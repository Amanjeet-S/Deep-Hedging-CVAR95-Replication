"""Conditional call deltas for the paper's risk-neutral GJR-GARCH model.

The terminal discontinuity is integrated analytically at one day.  For earlier
dates a deterministic Gaussian-quadrature dynamic program interpolates in log
moneyness and log conditional variance.  The delta is computed under the stock
numeraire: Delta=e^(-q*tau/252) P^S(S_T>K), which is exactly equivalent to
equation (9), but keeps the recursion bounded.  Under P^S the Q innovation is
N(sqrt(h), 1); the variance transition still uses z-eta from footnote 3.

This is a numerical alternative to the authors' nested 1,000-path Monte Carlo
benchmark, not an assertion that the paper used this algorithm.
"""

from __future__ import annotations

__author__ = "Amanjeet Singh"

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
from scipy.special import ndtr
from scipy.special import roots_hermitenorm


@dataclass(frozen=True)
class GJRParameters:
    mu: float
    omega: float
    alpha: float
    beta: float
    gamma: float
    r: float = 0.0167
    q: float = 0.0165
    trading_days: int = 252

    @classmethod
    def from_mapping(cls, values):
        aliases = {"a": "alpha", "b": "beta", "g": "gamma", "w": "omega",
                   "alpha[1]": "alpha", "beta[1]": "beta", "gamma[1]": "gamma"}
        fields = cls.__dataclass_fields__
        return cls(**{aliases.get(k, k): v for k, v in values.items()
                      if aliases.get(k, k) in fields})


@dataclass(frozen=True)
class GridConfig:
    x_min: float = -1.5
    x_max: float = 1.5
    nx: int = 4801
    logh_min: float = -12.0
    logh_max: float = -1.0
    nh: int = 221
    quadrature_nodes: int = 48
    dtype: str = "float32"


class DeltaGrid:
    """Build once, evaluate millions of conditional deltas by interpolation.

    The variance argument is the next-day conditional variance sigma_(t+1)^2,
    as specified in the paper's state, not the last observed squared shock.
    Arrays in ``probabilities`` have axes (remaining days, logh, logmoneyness).
    """

    def __init__(self, params: GJRParameters | dict, maturity_days: int = 63,
                 config: GridConfig | None = None):
        self.params = (params if isinstance(params, GJRParameters)
                       else GJRParameters.from_mapping(params))
        self.maturity_days = int(maturity_days)
        self.config = config or GridConfig()
        self.x = np.linspace(self.config.x_min, self.config.x_max, self.config.nx)
        self.logh = np.linspace(self.config.logh_min, self.config.logh_max,
                                self.config.nh)
        self.probabilities = None
        self.build_seconds = None
        self._eval_h_clamps = 0
        self._eval_x_clamps = 0
        self._eval_count = 0
        self.tail_grid = None
        self.tail_threshold = None

    @classmethod
    def build(cls, params: GJRParameters | dict, maturity_days: int = 63,
              config: GridConfig | None = None, progress=None):
        obj = cls(params, maturity_days, config)
        obj.compute(progress=progress)
        return obj

    def _transition(self, stock_measure=True):
        p, c = self.params, self.config
        nodes, weights = roots_hermitenorm(c.quadrature_nodes)
        weights = weights / np.sqrt(2 * np.pi)
        h = np.exp(self.logh)[:, None]
        sqrt_h = np.sqrt(h)
        carry = (p.r - p.q) / p.trading_days
        z = nodes[None, :] + (sqrt_h if stock_measure else 0)
        ret = carry - h / 2 + sqrt_h * z
        eta = (p.mu - carry + h / 2) / sqrt_h
        eps = z - eta
        hnext = p.omega + h * (p.alpha + p.gamma * (eps < 0)) * eps**2 + p.beta*h
        hi = (np.log(hnext) - self.logh[0]) / (self.logh[1]-self.logh[0])
        hi = np.clip(hi, 0, c.nh - 1)
        h0 = np.minimum(hi.astype(np.int32), c.nh - 2)
        hf = hi - h0
        dx = self.x[1] - self.x[0]
        xi = np.arange(c.nx)[None, :, None] + ret[:, None, :] / dx
        outside_low, outside_high = xi < 0, xi > c.nx - 1
        xi = np.clip(xi, 0, c.nx - 1)
        x0 = np.minimum(xi.astype(np.int32), c.nx - 2)
        xf = (xi - x0).astype(np.float32)
        return weights, h0, hf.astype(np.float32), x0, xf, outside_low, outside_high

    def _recursion(self, stock_measure, progress=None):
        c, p = self.config, self.params
        start = time.perf_counter()
        output = np.empty((self.maturity_days+1, c.nh, c.nx), dtype=c.dtype)
        output[0] = (self.x[None, :] > 0)
        h = np.exp(self.logh)[:, None]
        carry = (p.r-p.q)/p.trading_days
        drift = carry + (h/2 if stock_measure else -h/2)
        output[1] = ndtr((self.x[None, :] + drift) / np.sqrt(h))
        weights, h0, hf, x0, xf, low, high = self._transition(stock_measure)
        # Each quadrature node is vectorised over the complete two-dimensional
        # grid. Keeping this loop avoids huge simultaneous gather allocations.
        for tau in range(2, self.maturity_days+1):
            prev = output[tau-1]
            cur = np.zeros((c.nh, c.nx), dtype=np.float64)
            for k, w in enumerate(weights):
                a = h0[:, k, None]
                b = hf[:, k, None]
                ix = x0[:, :, k]
                fx = xf[:, :, k]
                lower = prev[a, ix] * (1-fx) + prev[a, ix+1] * fx
                upper = prev[a+1, ix] * (1-fx) + prev[a+1, ix+1] * fx
                interp = lower*(1-b) + upper*b
                interp[low[:, :, k]] = 0
                interp[high[:, :, k]] = 1
                cur += w * interp
            output[tau] = np.clip(cur, 0, 1)
            if progress and (tau % 10 == 0 or tau == self.maturity_days):
                progress(tau, time.perf_counter()-start)
        return output

    def compute(self, progress=None):
        start = time.perf_counter()
        self.probabilities = self._recursion(stock_measure=True, progress=progress)
        self.build_seconds = time.perf_counter()-start
        return self

    def _interpolate(self, surface, S, h, K):
        S, h = np.broadcast_arrays(np.asarray(S, dtype=float), np.asarray(h, dtype=float))
        x = np.log(S / K)
        lh = np.log(h)
        c = self.config
        self._eval_h_clamps += int(np.count_nonzero((lh < c.logh_min) | (lh > c.logh_max)))
        self._eval_x_clamps += int(np.count_nonzero((x < c.x_min) | (x > c.x_max)))
        self._eval_count += int(x.size)
        a = np.clip((lh-self.logh[0])/(self.logh[1]-self.logh[0]), 0, c.nh-1)
        b = np.clip((x-self.x[0])/(self.x[1]-self.x[0]), 0, c.nx-1)
        ia = np.minimum(a.astype(np.int32), c.nh-2)
        ib = np.minimum(b.astype(np.int32), c.nx-2)
        fa, fb = a-ia, b-ib
        result = ((1-fa)*((1-fb)*surface[ia,ib] + fb*surface[ia,ib+1])
                  + fa*((1-fb)*surface[ia+1,ib] + fb*surface[ia+1,ib+1]))
        result = np.where(x < c.x_min, 0, np.where(x > c.x_max, 1, result))
        return result

    def delta(self, tau, S, h, K=100.0):
        if self.probabilities is None:
            raise RuntimeError("Build the grid before evaluating delta")
        if not np.isscalar(tau):
            # Grouping by day makes vector-valued calls efficient.
            tau, S, h = np.broadcast_arrays(np.asarray(tau), S, h)
            result = np.empty(S.shape, dtype=float)
            for t in np.unique(tau):
                mask = tau == t
                result[mask] = self.delta(int(t), S[mask], h[mask], K)
            return result
        tau = int(tau)
        if not 0 <= tau <= self.maturity_days:
            raise ValueError("Remaining days are outside grid horizon")
        if tau == 1:
            p = self.params
            hh = np.asarray(h)
            return np.exp(-p.q/p.trading_days)*ndtr((
                np.log(np.asarray(S)/K)+(p.r-p.q)/p.trading_days+hh/2
            )/np.sqrt(hh))
        if tau == 0:
            return (np.asarray(S) > K).astype(float)
        result = np.exp(-self.params.q*tau/self.params.trading_days) * self._interpolate(
            self.probabilities[tau], S, h, K)
        if self.tail_grid is not None:
            ss, hh = np.broadcast_arrays(np.asarray(S), np.asarray(h))
            mask = hh > self.tail_threshold
            if np.any(mask):
                tail = self.tail_grid.delta(tau, ss[mask], hh[mask], K)
                result = np.array(result, copy=True)
                result[mask] = tail
        return result

    def enable_tail_refinement(self, threshold=0.01, progress=None):
        """Widen state coverage for exceptionally large conditional variance.

        Fine spot resolution is needed in ordinary low-volatility states. Large
        variance states instead need a much wider domain; a second grid trades
        spot resolution for coverage. It stays in RAM to keep disk use modest.
        """
        self.tail_threshold = threshold
        self.tail_grid = DeltaGrid.build(self.params, self.maturity_days,
            GridConfig(x_min=-4, x_max=4, nx=1601, logh_min=-12,
                       logh_max=3, nh=301, quadrature_nodes=48), progress)
        return self

    def call_prices(self, tau, S, h, K=100.0, q_probabilities=None):
        """Optional independently recursed exercise probabilities give prices.

        C=S Delta - K exp(-r*tau/252) Q(S_T>K). The Q recursion is built only
        if prices are requested, to avoid doubling benchmark build time.
        """
        if q_probabilities is None:
            q_probabilities = self._recursion(stock_measure=False)
        tau = int(tau)
        probq = self._interpolate(q_probabilities[tau], S, h, K)
        return np.asarray(S)*self.delta(tau, S, h, K) - K*np.exp(
            -self.params.r*tau/self.params.trading_days)*probq

    def save(self, path):
        if self.probabilities is None:
            raise RuntimeError("No grid to save")
        np.savez_compressed(path, probabilities=self.probabilities, x=self.x,
                            logh=self.logh, metadata=json.dumps({
                                "params": asdict(self.params), "config": asdict(self.config),
                                "maturity_days": self.maturity_days,
                                "build_seconds": self.build_seconds}))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            meta = json.loads(str(f["metadata"]))
            obj = cls(GJRParameters(**meta["params"]), meta["maturity_days"],
                      GridConfig(**meta["config"]))
            obj.probabilities = f["probabilities"]
            obj.build_seconds = meta.get("build_seconds")
            return obj

    def diagnostic_summary(self):
        return {"method": "stock-numeraire Gaussian quadrature with bilinear interpolation",
                "params": asdict(self.params), "config": asdict(self.config),
                "build_seconds": self.build_seconds, "evaluations": self._eval_count,
                "variance_clamps": self._eval_h_clamps, "moneyness_clamps": self._eval_x_clamps,
                "probability_min": float(self.probabilities.min()),
                "probability_max": float(self.probabilities.max()),
                "monotonicity_min_diff": float(np.diff(self.probabilities, axis=2).min()),
                "tail_refinement_threshold": self.tail_threshold,
                "tail_grid_config": asdict(self.tail_grid.config) if self.tail_grid else None}


def monte_carlo_delta(params: GJRParameters | dict, tau: int, S: float, h: float,
                      K=100.0, n_paths=1_000_000, seed=943, stock_measure=True):
    """Independent conditional simulation returning estimate and std error.

    Stock-measure simulation is an exact change of numeraire and avoids the
    unnecessarily variable e^(sum returns) pathwise weights of Q simulation.
    Set stock_measure=False to reproduce the paper's pathwise formula directly.
    One-day remaining expectations are integrated analytically in either case.
    """
    p = params if isinstance(params, GJRParameters) else GJRParameters.from_mapping(params)
    rng = np.random.default_rng(seed)
    carry = (p.r-p.q)/p.trading_days
    hh = np.full(n_paths, h, dtype=float)
    x = np.full(n_paths, np.log(S/K), dtype=float)
    absorbed = np.zeros(n_paths, dtype=bool)
    for t in range(max(0, tau-1)):
        if stock_measure:
            # Extremely large variances can overflow their quadratic update.
            # In this share-measure region the next normal exercise probability
            # exceeds 1-Phi(-25), and subsequent drift/variance growth keeps it
            # numerically indistinguishable from one. Record those paths rather
            # than permitting inf/inf to contaminate a validation average.
            absorbed |= (hh > 1e4) & (x > -hh/4)
            hh[absorbed] = h
            x[absorbed] = 0
        sh = np.sqrt(hh)
        z = rng.standard_normal(n_paths) + (sh if stock_measure else 0)
        ret = carry-hh/2+sh*z
        eta = (p.mu-carry+hh/2)/sh
        eps = z-eta
        x += ret
        hh = p.omega + (p.alpha+p.gamma*(eps < 0))*hh*eps**2+p.beta*hh
    if tau == 0:
        vals = (x > 0).astype(float)
    elif stock_measure:
        vals = ndtr((x+carry+hh/2)/np.sqrt(hh)) * np.exp(-p.q*tau/p.trading_days)
        vals[absorbed] = np.exp(-p.q*tau/p.trading_days)
    else:
        # Current S_T-1 / S_0 multiplied by the exact last-step weighted payoff.
        vals = (np.exp(x-np.log(S/K)) * np.exp(carry)
                * ndtr((x+carry+hh/2)/np.sqrt(hh))
                * np.exp(-p.r*tau/p.trading_days))
    return {"estimate": float(vals.mean()),
            "standard_error": float(vals.std(ddof=1)/np.sqrt(n_paths)),
            "n_paths": n_paths, "seed": seed, "stock_measure": stock_measure,
            "absorbed_extreme_variance_paths": int(absorbed.sum())}


def validate_grid(grid: DeltaGrid, states, n_paths=500_000, seed=4307):
    output=[]
    for j, state in enumerate(states):
        tau, S, h = state
        d = float(grid.delta(tau,S,h))
        mc=monte_carlo_delta(grid.params,tau,S,h,n_paths=n_paths,seed=seed+j)
        output.append({"tau":tau,"S":S,"h":h,"grid_delta":d,**mc,
                       "grid_minus_mc":d-mc["estimate"],
                       "difference_in_mc_standard_errors":(d-mc["estimate"])/mc["standard_error"]
                       if mc["standard_error"] else None})
    return output
