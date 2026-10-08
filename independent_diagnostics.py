"""Likelihood multistart and analytical physical-variance moment diagnostics."""
from __future__ import annotations

__author__ = "Amanjeet Singh"

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import arch
from arch import arch_model
import numpy as np
import pandas as pd
import scipy


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def multistart(root, folder, calibration):
    data = pd.read_csv(root / "data/SP500_random.csv", index_col=0).iloc[5031:, :]
    returns = np.log(data.Close / data.Close.shift(1)).dropna()
    if len(returns) != 1259:
        raise AssertionError("The retained historical sample must contain 1,259 returns")
    p = calibration["selected_parameters"]
    retained_start = np.array([100*p["mu"], 10000*p["omega"], p["alpha"], p["gamma"], p["beta"]])
    printed_start = np.array([.06, 10000*p["omega"], .11, .20, .78])
    starts = [
        ("library_default", None),
        ("retained_decimal_fit_converted_to_percentage_units", retained_start),
        ("printed_coefficient_hybrid_in_percentage_units", printed_start),
        ("diverse_lower_persistence", np.array([.04, .08, .15, .10, .55])),
        ("diverse_higher_persistence", np.array([.10, .02, .05, .10, .86])),
    ]
    stored = calibration["scaled_conditioning_control"]
    records = []
    for label, start in starts:
        if start is not None:
            _, omega, alpha, gamma, beta = start
            if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha+gamma >= 0 and alpha+gamma/2+beta < 1):
                raise AssertionError("A supplied start violates the GJR constraints")
        model = arch_model(100*returns, vol="Garch", p=1, o=1, q=1, rescale=False)
        begun = time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = model.fit(starting_values=start, update_freq=0, disp="off",
                               options={"ftol": 1e-10, "maxiter": 1000})
        raw = {key: float(value) for key, value in fitted.params.items()}
        decimal = {"mu": raw["mu"]/100, "omega": raw["omega"]/10000,
                   "alpha": raw["alpha[1]"], "gamma": raw["gamma[1]"], "beta": raw["beta[1]"]}
        h0 = float(fitted.conditional_volatility.iloc[-1]**2/10000)
        residual = float(fitted.resid.iloc[-1]/100)
        forecast = decimal["omega"] + (decimal["alpha"]+decimal["gamma"]*(residual < 0))*residual**2 + decimal["beta"]*h0
        converted_ll = float(fitted.loglikelihood + len(returns)*math.log(100))
        rejected_start = any(type(item.message).__name__ == "StartingValueWarning" for item in captured)
        record = {
            "label": label,
            "supplied_start_percentage_units": None if start is None else start.tolist(),
            "parameter_order": ["mu", "omega", "alpha", "gamma", "beta"],
            "supplied_start_accepted": None if start is None else not rejected_start,
            "params_percentage_units": raw,
            "params_decimal_returns": decimal,
            "loglikelihood_percentage_units": float(fitted.loglikelihood),
            "loglikelihood_decimal_units": converted_ll,
            "difference_from_stored_default_scaled_loglikelihood": converted_ll-stored["loglikelihood_in_decimal_return_units"],
            "terminal_fitted_variance": h0,
            "observed_residual_one_step_forecast": float(forecast),
            "last_fitted_residual": residual,
            "persistence": decimal["alpha"]+decimal["gamma"]/2+decimal["beta"],
            "success": bool(fitted.optimization_result.success),
            "convergence_flag": int(fitted.convergence_flag),
            "iterations": int(fitted.optimization_result.nit),
            "function_evaluations": int(fitted.optimization_result.nfev),
            "message": str(fitted.optimization_result.message),
            "warnings": [str(item.message) for item in captured],
            "seconds": time.monotonic()-begun,
        }
        records.append(record)
        print(f"{label}: LL(decimal)={converted_ll:.12f}; success={record['success']}; iterations={record['iterations']}", flush=True)
    likelihoods = [record["loglikelihood_decimal_units"] for record in records]
    parameter_names = ["mu", "omega", "alpha", "gamma", "beta"]
    record = {
        "author": __author__,
        "method": "Constant-mean Gaussian GJR-GARCH(1,1), percentage returns, ARCH constrained SLSQP",
        "data": "data/SP500_random.csv", "sample_count": len(returns),
        "first_return_date": str(returns.index[0]), "last_return_date": str(returns.index[-1]),
        "optimizer_options": {"ftol": 1e-10, "maxiter": 1000},
        "versions": {"python": platform.python_version(), "arch": arch.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
        "stored_default_scaled_fit": stored,
        "fits": records,
        "summary": {
            "all_successful": all(item["success"] for item in records),
            "all_supplied_starts_accepted": all(item["supplied_start_accepted"] is not False for item in records),
            "queried_loglikelihood_range": [min(likelihoods), max(likelihoods)],
            "queried_loglikelihood_spread": max(likelihoods)-min(likelihoods),
            "largest_queried_loglikelihood_label": records[int(np.argmax(likelihoods))]["label"],
            "parameter_spreads_decimal_units": {name: max(item["params_decimal_returns"][name] for item in records)-min(item["params_decimal_returns"][name] for item in records) for name in parameter_names},
        },
        "interpretation": "Consistency across these starts is evidence about the queried likelihood solutions, not proof of a global maximum or recovery of the paper's calibration. No fitted candidate is selected by a published hedging statistic, and retained experiment parameters are unchanged.",
        "relative_risk_formula_evidence": {
            "source": "src/visualization/strategy_evaluation.py:219-234",
            "formula": "cvar_bs=mean(sorted(delta_loss)[floor(a*n):]); cvar_dh=mean(sorted(deep_loss)[floor(a*n):]); Relative_CVaR=cvar_dh/cvar_bs",
            "historical_csv_field": "Relative_metric",
            "qualification": "The current cached writer explicitly defines a ratio under the name Relative_CVaR. The stored CSV uses a different header, Relative_metric; its mapping to this writer is not recorded. Any gap inferred from that historical column is conditional on this header mapping, not a directly stored or published gap.",
        },
    }
    write_json(folder / "multistart_fits.json", record)
    table = []
    for item in records:
        table.append({"author": __author__, "label": item["label"], **item["params_decimal_returns"],
                      "loglikelihood_decimal_units": item["loglikelihood_decimal_units"],
                      "terminal_fitted_variance": item["terminal_fitted_variance"],
                      "observed_residual_forecast": item["observed_residual_one_step_forecast"],
                      "success": item["success"], "iterations": item["iterations"]})
    pd.DataFrame(table).to_csv(folder / "multistart_fits.csv", index=False)
    return record


def variance_moments(folder, calibration):
    baseline = dict(calibration["selected_parameters"])
    observed = dict(baseline, initial_variance=calibration["authors_unscaled_fit"]["next_variance_forecast_from_observed_last_residual"])
    compatible = dict(baseline, initial_variance=7.75e-5)
    scaled_fit = calibration["scaled_conditioning_control"]
    scaled_raw = scaled_fit["params_decimal_returns"]
    scaled = {"mu": scaled_raw["mu"], "omega": scaled_raw["omega"], "alpha": scaled_raw["alpha[1]"],
              "gamma": scaled_raw["gamma[1]"], "beta": scaled_raw["beta[1]"],
              "initial_variance": scaled_fit["initial_fitted_variance_h0"]}
    scenarios = [("baseline_fitted_state", baseline), ("observed_last_residual_forecast", observed),
                 ("illustrative_price_compatible_initial_variance", compatible),
                 ("percentage_return_conditioning_fit_terminal_state", scaled)]
    summaries, rows, sequences = {}, [], {}
    for label, p in scenarios:
        omega, alpha, gamma, beta, h0 = (p[name] for name in ("omega", "alpha", "gamma", "beta", "initial_variance"))
        psi = beta+alpha+gamma/2
        chi = beta**2 + 2*beta*(alpha+gamma/2) + 3*(alpha**2+alpha*gamma+gamma**2/2)
        m1, m2 = h0, h0**2
        mean_sequence, second_sequence, formula_sequence = [], [], []
        for t in range(64):
            formula = h0+t*omega if psi == 1 else psi**t*h0 + omega*(1-psi**t)/(1-psi)
            mean_sequence.append(m1)
            second_sequence.append(m2)
            formula_sequence.append(formula)
            if t < 63:
                rows.append({"author": __author__, "scenario": label, "decision_date": t,
                             "return_date": t+1, "mean_variance": m1, "second_variance_moment": m2,
                             "variance_of_conditional_variance": max(m2-m1*m1, 0),
                             "closed_form_mean_variance": formula})
            old_m1 = m1
            m1 = omega+psi*old_m1
            m2 = omega**2 + 2*omega*psi*old_m1 + chi*m2
        mean_sequence, second_sequence = np.asarray(mean_sequence), np.asarray(second_sequence)
        error = float(np.max(np.abs(mean_sequence-np.asarray(formula_sequence))))
        if error > 1e-16:
            raise AssertionError("Closed-form mean variance disagrees with recursion")
        average = float(mean_sequence[:63].mean())
        summaries[label] = {
            "parameters": p, "psi": psi, "chi": chi,
            "mean_variance_over_63_returns": average,
            "annualised_rms_volatility": math.sqrt(252*average),
            "mean_variance_at_date63": float(mean_sequence[63]),
            "second_variance_moment_at_date63": float(second_sequence[63]),
            "closed_form_mean_max_error": error,
            "stationary_mean_variance_if_psi_less_than_one": omega/(1-psi) if psi < 1 else None,
            "finite_stationary_second_moment_condition_met": chi < 1,
        }
        sequences[label] = mean_sequence
    initial_coupling = {}
    psi = summaries["baseline_fitted_state"]["psi"]
    for label in ("observed_last_residual_forecast", "illustrative_price_compatible_initial_variance"):
        delta_h0 = summaries[label]["parameters"]["initial_variance"]-baseline["initial_variance"]
        actual = sequences[label]-sequences["baseline_fitted_state"]
        predicted = psi**np.arange(64)*delta_h0
        error = float(np.max(np.abs(actual-predicted)))
        if error > 1e-16:
            raise AssertionError("Coupled initial-variance mean difference disagrees")
        initial_coupling[label] = {"initial_variance_difference": delta_h0,
                                  "max_error_in_psi_power_difference": error,
                                  "mean_variance_difference_over_63_returns": float(actual[:63].mean()),
                                  "date63_mean_variance_difference": float(actual[63])}
    record = {
        "author": __author__, "measure": "physical",
        "dates": "Decision dates 0 through 62 supply the variances for returns 1 through 63; date63 is a post-horizon diagnostic.",
        "moment_derivation": {
            "A": "beta+(alpha+gamma*1{Z<0})*Z^2, Z standard normal independent of current variance",
            "normal_moments": "E[Z^2]=1; E[Z^2*1{Z<0}]=1/2; E[Z^4]=3; E[Z^4*1{Z<0}]=3/2",
            "psi": "E[A]=beta+alpha+gamma/2",
            "chi": "E[A^2]=beta^2+2*beta*(alpha+gamma/2)+3*(alpha^2+alpha*gamma+gamma^2/2)",
            "m1": "m1_next=omega+psi*m1; m1_0=h0",
            "m2": "m2_next=omega^2+2*omega*psi*m1+chi*m2; m2_0=h0^2",
            "closed_form_m1": "m1_t=psi^t*h0+omega*(1-psi^t)/(1-psi), with m1_t=h0+t*omega when psi=1",
            "coupled_mean_difference": "For identical coefficients and initial-state difference delta_h0, delta_m1_t=psi^t*delta_h0",
            "rms": "annualised_RMS_volatility=sqrt(252*mean(m1_0,...,m1_62))",
        },
        "scenarios": summaries, "initial_state_checks": initial_coupling,
        "csv": "physical_variance_moments.csv",
        "interpretation": "These are exact physical conditional-variance moment recursions for deterministic initial variance, not risk-neutral implied volatilities or proofs of hedge-loss moments. Finite-horizon variance moments can exist even when stock or option-payoff moments fail. A chi at least one rules out a finite stationary variance second moment; it does not make these finite-date recursions infinite.",
    }
    write_json(folder / "physical_variance_moments.json", record)
    pd.DataFrame(rows).to_csv(folder / "physical_variance_moments.csv", index=False)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, folder = args.root.expanduser().resolve(), args.folder.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    calibration_path = root / "calibration.json"
    original = calibration_path.read_bytes()
    calibration = json.loads(original)
    fits = multistart(root, folder, calibration)
    moments = variance_moments(folder, calibration)
    if calibration_path.read_bytes() != original:
        raise AssertionError("The retained calibration changed during these independent diagnostics")
    print("All supplied starts accepted:", fits["summary"]["all_supplied_starts_accepted"])
    for label, summary in moments["scenarios"].items():
        print(f"{label}: psi={summary['psi']:.9f}, chi={summary['chi']:.9f}, annualised RMS volatility={summary['annualised_rms_volatility']:.9f}")
    print("Independent likelihood and physical-variance diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
