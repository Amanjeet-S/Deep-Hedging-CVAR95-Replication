"""Independent replication of the 95% CVaR deep-hedging experiment.

Paper equations, corrected next-day variance timing, PyTorch training.
No historical observations or published numbers are used as training targets.
"""
from __future__ import annotations

__author__ = "Amanjeet Singh"

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    maturity: int = 63
    spot: float = 100.0
    strike: float = 100.0
    r: float = 0.0167
    q: float = 0.0165
    confidence: float = 0.95
    borrowing_limit: float = 100.0
    training_paths: int = 400000
    validation_paths: int = 50000
    test_paths: int = 100000
    epochs: int = 50
    batch_size: int = 1000
    learning_rate: float = 0.0005
    training_seed: int = 43
    validation_seed: int = 44
    test_seed: int = 45
    model_seed: int = 4301
    price_seed: int = 46
    initial_capital: float = 3.16


def physical_paths(n: int, params: dict, cfg: Config, seed: int):
    """h_t here means the previsible variance of return t+1, exactly Eq.(7)."""
    rng = np.random.RandomState(seed)
    log_s = np.full(n, np.log(cfg.spot), dtype=np.float64)
    h = np.full(n, params['initial_variance'], dtype=np.float64)
    features = np.empty((n, cfg.maturity, 3), dtype=np.float32)
    gains = np.empty((n, cfg.maturity), dtype=np.float32)
    er, eq = np.exp(cfg.r/252), np.exp(cfg.q/252)
    for t in range(cfg.maturity):
        s = np.exp(log_s)
        features[:, t, 0] = log_s
        features[:, t, 1] = (cfg.maturity-t)/252
        features[:, t, 2] = np.sqrt(252*h)
        z = rng.normal(size=n)
        log_s += params['mu'] + np.sqrt(h)*z
        sn = np.exp(log_s)
        gains[:, t] = eq*sn-er*s
        h = params['omega'] + h*(params['alpha']+params['gamma']*(z<0))*z*z + params['beta']*h
    return features, gains, np.maximum(np.exp(log_s)-cfg.strike, 0).astype(np.float32)


def risk_neutral_price(params: dict, cfg: Config, n=2000000):
    """Antithetic Q pricing through a bounded put and put-call parity.

    Simulate the first T-1 periods, then integrate the final Gaussian shock.
    A path remains in the sample even if its positive spot underflows to zero:
    the put contribution becomes the discounted strike only after actual
    exp(log_s) == 0. No finite positive variance is clipped. Conditional puts
    are bounded by the discounted strike, ensuring a finite-variance price
    estimator even when direct call-payoff second moments are infinite.
    """
    from scipy.special import ndtr, log_ndtr

    if n <= 0 or n % 2:
        raise ValueError('Antithetic pricing requires a positive even path count')
    if cfg.maturity < 1:
        raise ValueError('Option maturity must be at least one period')
    rng = np.random.RandomState(cfg.price_seed)
    count, total, total2 = 0, 0.0, 0.0
    stock_total = 0.0
    absorbed_stock_zero = 0
    max_live_variance = 0.0
    put_sample_min, put_sample_max = np.inf, -np.inf
    put_bound_violations = 0
    log_smallest_positive = math.log(np.nextafter(0.0, 1.0))
    dt = 1.0/252
    discounted_put_bound = cfg.strike*np.exp(-cfg.r*cfg.maturity*dt)
    parity_term = cfg.spot*np.exp(-cfg.q*cfg.maturity*dt)-discounted_put_bound
    for start in range(0, n, 100000):
        m = min(100000, n-start)
        half = m//2
        log_s = np.full(m, np.log(cfg.spot))
        h = np.full(m, params['initial_variance'])
        alive = np.ones(m, dtype=bool)
        for t in range(cfg.maturity-1):
            zz = rng.normal(size=half)
            z = np.concatenate((zz, -zz))
            ix = np.flatnonzero(alive)
            hv, zv = h[ix], z[ix]
            eta = (params['mu']-(cfg.r-cfg.q)*dt+hv/2)/np.sqrt(hv)
            shifted = zv-eta
            log_s[ix] += (cfg.r-cfg.q)*dt-hv/2+np.sqrt(hv)*zv
            # Restrict exp checks to candidates below the smallest positive
            # representable number, then require actual floating-point zero.
            zero = np.zeros(len(ix), dtype=bool)
            candidates = log_s[ix] < log_smallest_positive
            zero[candidates] = np.exp(log_s[ix[candidates]]) == 0.0
            absorbed_stock_zero += int(zero.sum())
            alive[ix[zero]] = False
            log_s[ix[zero]] = -np.inf
            keep = ~zero
            ki = ix[keep]
            if len(ki):
                hk, sk = hv[keep], shifted[keep]
                h[ki] = params['omega']+hk*(params['alpha']+params['gamma']*(sk<0))*sk**2+params['beta']*hk
                if not np.all(np.isfinite(h[ki])):
                    raise FloatingPointError('Variance overflow for a positive representable spot')
                max_live_variance = max(max_live_variance, float(h[ki].max()))
        # At T-1 the next-return variance is known. The one-period conditional
        # put price integrates the last shock and already discounts one day.
        # An absorbed zero-spot path has put value equal to the discounted
        # strike; it is retained with that contribution, not discarded.
        values = np.full(m, discounted_put_bound)
        ix = np.flatnonzero(alive)
        root_h = np.sqrt(h[ix])
        d1 = (log_s[ix]-np.log(cfg.strike)+(cfg.r-cfg.q)*dt+h[ix]/2)/root_h
        d2 = d1-root_h
        spot = np.exp(log_s[ix])
        # log_ndtr avoids an overflow-times-zero indeterminacy in the bounded
        # asset-binary put term for a very large positive log spot.
        asset_binary_put = np.exp(log_s[ix]-cfg.q*dt+log_ndtr(-d1))
        one_period_put = cfg.strike*np.exp(-cfg.r*dt)*ndtr(-d2)-asset_binary_put
        values[ix] = np.exp(-cfg.r*(cfg.maturity-1)*dt)*np.maximum(one_period_put, 0.0)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError('Nonfinite conditional put price; paths are not discarded')
        put_sample_min = min(put_sample_min, float(values.min()))
        put_sample_max = max(put_sample_max, float(values.max()))
        put_bound_violations += int(((values < -1e-12) | (values > discounted_put_bound+1e-12)).sum())
        if put_bound_violations:
            raise FloatingPointError('Conditional put violates its deterministic bounds')
        paired = (values[:half]+values[half:])/2
        count += len(paired)
        total += float(paired.sum())
        total2 += float(np.square(paired).sum())
        stock_total += float(spot.sum()*np.exp((cfg.r-cfg.q)*dt))
    put_mean = total/count
    se = np.sqrt(max((total2/count-put_mean*put_mean)/(count-1), 0.0)) if count > 1 else 0.0
    return {'price':parity_term+put_mean, 'standard_error':float(se), 'paths':n,
            'put_price':put_mean,
            'parity_term':float(parity_term),
            'estimator_note':'Bounded final-period conditional put plus put-call parity; price SE equals antithetic put-pair SE',
            'bound_stats':{'lower':0.0, 'upper':float(discounted_put_bound),
                           'sample_min':put_sample_min, 'sample_max':put_sample_max,
                           'violations':put_bound_violations},
            'antithetic_pairs':count,
            'discounted_dividend_stock_mean':stock_total/n*np.exp(-(cfg.r-cfg.q)*cfg.maturity*dt),
            'final_period_integrated_analytically':True,
            'simulated_periods':cfg.maturity-1,
            'absorbed_numerically_zero_stock_paths':absorbed_stock_zero,
            'paths_dropped':0,
            'variance_clipped':False,
            'max_live_variance':max_live_variance}


class Hedger(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        width = 4
        for _ in range(4):
            layers += [nn.Linear(width,56),nn.ReLU()]
            width = 56
        layers.append(nn.Linear(56,1))
        self.policy = nn.Sequential(*layers)
        self.cash_growth = math.exp(cfg.r/252)
        self.borrowing_limit = cfg.borrowing_limit
        for layer in self.policy:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor, gains: torch.Tensor, initial: float):
        v = torch.full_like(features[:,0,0],initial)
        for t in range(features.shape[1]):
            state = torch.cat((features[:,t,:],v.unsqueeze(1)),dim=1)
            raw = self.policy(state).squeeze(1)
            cap = (v+self.borrowing_limit)/torch.exp(features[:,t,0])
            delta = torch.minimum(raw,cap)
            v = self.cash_growth*v+delta*gains[:,t]
        return v

    @torch.jit.export
    def positions(self, features: torch.Tensor, gains: torch.Tensor, initial: float):
        v = torch.full_like(features[:,0,0],initial)
        deltas = torch.jit.annotate(list[torch.Tensor],[])
        min_cash = torch.full_like(v,1e20)
        for t in range(features.shape[1]):
            state = torch.cat((features[:,t,:],v.unsqueeze(1)),dim=1)
            raw = self.policy(state).squeeze(1)
            s = torch.exp(features[:,t,0])
            delta = torch.minimum(raw,(v+self.borrowing_limit)/s)
            deltas.append(delta)
            min_cash = torch.minimum(min_cash,v-delta*s)
            v = self.cash_growth*v+delta*gains[:,t]
        return v, torch.stack(deltas,dim=1), min_cash


def cvar(losses,alpha=.95):
    x = np.asarray(losses,dtype=np.float64)
    k = int(math.floor(alpha*len(x)))
    return float(np.partition(x,k)[k:].mean())


def cvar_influence(losses, alpha=.95):
    x = np.asarray(losses,dtype=np.float64)
    var = float(np.partition(x,int(alpha*len(x)))[int(alpha*len(x))])
    influence = np.maximum(x-var,0)/(1-alpha)
    return influence-influence.mean()


def configure_torch(threads=1):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def train(params: dict, cfg: Config, folder: Path, device='cpu', threads=1):
    configure_torch(threads)
    torch.manual_seed(cfg.model_seed)
    folder.mkdir(parents=True,exist_ok=True)
    model = torch.jit.script(Hedger(cfg).to(device))
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,eps=1e-8)
    print('Generating independent training and validation samples',flush=True)
    f,g,pay = physical_paths(cfg.training_paths,params,cfg,cfg.training_seed)
    fv,gv,pv = physical_paths(cfg.validation_paths,params,cfg,cfg.validation_seed)
    f,g,pay = torch.from_numpy(f),torch.from_numpy(g),torch.from_numpy(pay)
    fv,gv,pv = torch.from_numpy(fv),torch.from_numpy(gv),torch.from_numpy(pv)
    rng = np.random.RandomState(cfg.training_seed+1000)
    best = float('inf')
    history=[]
    started=time.monotonic()
    tail=cfg.batch_size-int(cfg.confidence*cfg.batch_size)
    for epoch in range(1,cfg.epochs+1):
        tic=time.monotonic()
        indices=rng.permutation(cfg.training_paths)
        training_losses=[]
        model.train()
        for j in range(0,cfg.training_paths,cfg.batch_size):
            ids=indices[j:j+cfg.batch_size]
            fb,gb,pb=f[ids].to(device),g[ids].to(device),pay[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            losses=pb-model(fb,gb,cfg.initial_capital)
            objective=torch.topk(losses,tail).values.mean()
            objective.backward()
            optimizer.step()
            training_losses.append(losses.detach().cpu().numpy())
        model.eval()
        validation_losses=[]
        with torch.no_grad():
            for j in range(0,cfg.validation_paths,cfg.batch_size):
                losses=pv[j:j+cfg.batch_size].to(device)-model(fv[j:j+cfg.batch_size].to(device),gv[j:j+cfg.batch_size].to(device),cfg.initial_capital)
                validation_losses.append(losses.cpu().numpy())
        val=cvar(np.concatenate(validation_losses),cfg.confidence)
        tr=cvar(np.concatenate(training_losses),cfg.confidence)
        if val<best:
            best=val
            torch.save({'state_dict':model.state_dict(),'config':asdict(cfg),'parameters':params,
                        'epoch':epoch,'validation_cvar':val},folder/'agent95.pt')
        row={'epoch':epoch,'training_cvar':tr,'validation_cvar':val,
             'best_validation_cvar':best,'seconds':time.monotonic()-tic}
        history.append(row)
        (folder/'training_history.json').write_text(json.dumps(history,indent=2))
        print(f"Epoch {epoch:02d}/{cfg.epochs} train={tr:.5f} validation={val:.5f} best={best:.5f} seconds={row['seconds']:.1f}",flush=True)
    print(f'Training finished in {time.monotonic()-started:.1f}s',flush=True)
    return history


def benchmark(params,cfg):
    f,g,p=physical_paths(1000,params,cfg,123)
    configure_torch(1)
    out={}
    for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
        torch.manual_seed(cfg.model_seed)
        model=torch.jit.script(Hedger(cfg).to(device))
        opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
        fb,gb,pb=torch.from_numpy(f).to(device),torch.from_numpy(g).to(device),torch.from_numpy(p).to(device)
        times=[]
        for k in range(5):
            tic=time.monotonic()
            opt.zero_grad(set_to_none=True)
            loss=torch.topk(pb-model(fb,gb,cfg.initial_capital),50).values.mean()
            loss.backward();opt.step()
            if device=='mps':torch.mps.synchronize()
            times.append(time.monotonic()-tic)
        out[device]=float(np.mean(times[2:]))
        print(device,times,flush=True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--calibration',default='calibration.json')
    ap.add_argument('--folder',default='run95')
    ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--epochs',type=int,default=50)
    args=ap.parse_args()
    calibration=json.loads(Path(args.calibration).read_text())
    params=calibration['selected_parameters']
    cfg=Config(epochs=args.epochs)
    if args.benchmark:
        print(json.dumps(benchmark(params,cfg),indent=2))
    else:
        train(params,cfg,Path(args.folder),args.device)
