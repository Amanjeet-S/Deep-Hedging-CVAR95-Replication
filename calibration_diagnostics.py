"""Prespecified calibration and initial-variance price diagnostics."""
from __future__ import annotations

__author__ = "Amanjeet Singh"

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


def load_module(root):
    specification = importlib.util.spec_from_file_location("replication", root / "replication.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules["replication"] = module
    specification.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def write(output, data):
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(output)


def historical_audit(root, authors, calibration):
    raw = pd.read_csv(root / "data/SP500_random.csv", index_col=0)
    sample = raw.iloc[5031:, :]
    returns = np.log(sample.Close / sample.Close.shift(1)).dropna()
    p = calibration["selected_parameters"]
    residual = float(returns.iloc[-1] - p["mu"])
    forecast = (p["omega"] + (p["alpha"] + p["gamma"] * (residual < 0)) * residual ** 2
                + p["beta"] * p["initial_variance"])
    expected_random_first_variance = p["omega"] + (
        p["alpha"] + p["gamma"] / 2 + p["beta"]) * p["initial_variance"]
    record = {
        "author": __author__,
        "paper_reference": {
            "authors": ["Pascal Fran\u00e7ois", "Genevi\u00e8ve Gauthier", "Fr\u00e9d\u00e9ric Godin", "Carlos Octavio P\u00e9rez Mendoza"],
            "year": 2025,
            "title": "Is the difference between deep hedging and delta hedging a statistical arbitrage?",
            "journal": "Finance Research Letters", "volume": 73, "article": 106590,
            "doi": "10.1016/j.frl.2024.106590",
        },
        "historical_sample": {
            "dataset": "data/SP500_random.csv", "raw_observations": len(raw),
            "slice_zero_based_start_row": 5031,
            "preceding_close_date": str(sample.index[0]),
            "first_return_date": str(returns.index[0]), "last_return_date": str(returns.index[-1]),
            "return_count": len(returns), "return_definition": "log(Close_t/Close_(t-1)) in decimal units",
            "source_fit_call": "arch_model(log_returns,vol='Garch',p=1,o=1,q=1,rescale=False).fit(disp='off')",
            "units": {"mu": "daily decimal log return", "omega": "daily decimal log-return variance",
                      "alpha_gamma_beta": "dimensionless"},
        },
        "initial_state": {
            "terminal_fitted_variance": p["initial_variance"],
            "last_observed_log_return": float(returns.iloc[-1]), "last_fitted_residual": residual,
            "observed_residual_one_step_forecast": forecast,
            "forecast_minus_terminal": forecast - p["initial_variance"],
            "forecast_relative_change": forecast / p["initial_variance"] - 1,
            "forecast_formula": "omega+(alpha+gamma*1{residual<0})*residual^2+beta*h_terminal",
            "cached_source_first_return": "Source generates an artificial residual sqrt(h_terminal)*Z0 and updates variance before the first simulated return; it does not use the final observed fitted residual.",
            "cached_source_physical_first_return_variance_mean": expected_random_first_variance,
            "source_random_state_warning": "The source first-return variance is random across paths. A deterministic observed-residual forecast does not reproduce this convention.",
            "source_feature_timing": "The saved first decision volatility is the terminal fitted variance, while the first generated return uses the variance after the artificial residual update.",
            "independent_run_convention": "Terminal fitted variance is used as the known first-return variance, without an artificial preliminary innovation.",
        },
        "paper_disclosure": {
            "sample_dates": ["4 January 2016", "31 December 2020"],
            "printed_parameters": {"mu": "0.06%", "omega": "0.01%", "alpha": .11, "gamma": .20, "beta": .78},
            "printed_rates": {"r": .0167, "q": .0165}, "reported_initial_option_price": 3.16,
            "initial_variance_disclosed": False, "full_precision_parameters_disclosed": False,
            "intercept_units_warning": "The printed 0.01% does not identify the units and scaling needed to recover the fitted daily variance intercept.",
        },
        "conditioning": {
            "retained_decimal_loglikelihood": calibration["authors_unscaled_fit"]["loglikelihood_in_decimal_return_units"],
            "percentage_fit_decimal_loglikelihood": calibration["scaled_conditioning_control"]["loglikelihood_in_decimal_return_units"],
            "unit_conversion": "mu_decimal=mu_percentage/100; omega_decimal=omega_percentage/10000; dimensionless coefficients unchanged",
            "likelihood_conversion": "LL_decimal=LL_percentage+n_returns*log(100)",
            "conclusion": "The larger scaled-fit likelihood demonstrates conditioning sensitivity; optimizer success alone does not establish the global maximum.",
        },
        "cached_source_evidence": {
            "fit": "src/features/features_simulation.py:119-175",
            "artificial_initial_residual": "src/features/features_simulation.py:176-182",
            "variance_before_return": "src/features/features_simulation.py:77-90",
            "lagged_volatility_feature": "src/features/features_simulation.py:188-190",
            "risk_neutral_initial_residual": "src/utils.py:62-80",
            "source_rate_defaults": {"r": .026623194, "q": .01772245, "model": "Black-Scholes"},
            "source_pricing_issues": "The Q utility hardcodes q and discounts its already daily risk-free rate by 252 again.",
        },
    }
    if len(returns) != calibration["n_returns"] or forecast != calibration["authors_unscaled_fit"]["next_variance_forecast_from_observed_last_residual"]:
        raise AssertionError("Retained historical sample or observed-residual forecast does not agree")
    if authors is not None:
        files = [path for path in authors.rglob("*") if path.is_file() and ".git" not in path.relative_to(authors).parts]
        substantive = [path for path in files if path.name not in (".gitignore", ".gitkeep")]
        model_files = [str(path.relative_to(authors)) for path in substantive if path.relative_to(authors).parts[0] == "models"]
        data_files = [str(path.relative_to(authors)) for path in substantive if path.relative_to(authors).parts[0] == "data"]
        archives = [str(path.relative_to(authors)) for path in substantive if path.suffix.lower() in (".zip", ".gz", ".tar", ".7z", ".npz")]
        notebook = read(authors / "notebooks/deep_hedging_pipeline.ipynb")
        output = "".join("".join(item.get("text", [])) for cell in notebook.get("cells", []) for item in cell.get("outputs", []))
        requirements = (authors / "requirements.txt").read_text()
        shallow = subprocess.run(["git", "-C", str(authors), "rev-parse", "--is-shallow-repository"],
                                 capture_output=True, text=True, check=True).stdout.strip() == "true"
        commit_count = int(subprocess.run(["git", "-C", str(authors), "rev-list", "--count", "--all"],
                                         capture_output=True, text=True, check=True).stdout)
        metrics_name = "data/results/Training/Random_63/Call/ATM/plots/metrics.csv"
        metrics_path = authors / metrics_name
        aggregate = []
        if metrics_path.is_file():
            table = pd.read_csv(metrics_path, index_col=0)
            for _, row in table.loc[table["Market"] == "GJR-GARCH"].iterrows():
                deep_cvar = float(row["Metric_dh"])
                ratio = float(row["Relative_metric"])
                aggregate.append({"agent": row["Strategy"], "option_price": float(row["Option_price"]),
                                  "deep_cvar": deep_cvar, "stored_relative_metric": ratio,
                                  "ratio_mapping_confirmed": False,
                                  "conditionally_inferred_delta_cvar": deep_cvar / ratio,
                                  "conditionally_inferred_cvar_gap": deep_cvar - deep_cvar / ratio,
                                  "overlay_cvar": float(row["risk_metric_(delta_difference)"]),
                                  "overlay_mean": float(row["Mean_(delta_difference)"])})
        record["cached_source_inventory"] = {
            "substantive_files": len(substantive), "placeholder_files": len(files) - len(substantive),
            "model_checkpoint_files": model_files, "data_files": data_files, "archive_files": archives,
            "published_table_gjr_results_or_full_precision_parameters_stored": False,
            "gjr_aggregate_results_file": metrics_name if aggregate else None,
            "gjr_aggregate_results": aggregate,
            "aggregate_results_warning": "These GJR aggregate rows differ from the published table and contain no fitted coefficients or initial variance. The current writer defines a CVaR ratio, but uses the different header Relative_CVaR; the archived Relative_metric header mapping is not recorded. Delta risks and gaps inferred from this field are conditional on that mapping. Only directly stored values are used in the source comparison.",
            "stored_notebook_experiment": "Black-Scholes" if "Black-Scholes" in output and "GJR-GARCH" not in output else "Inspect notebook outputs",
            "stored_notebook_option_price": 3.945866 if "3.945866" in output else None,
            "notebook_python_version": notebook.get("metadata", {}).get("language_info", {}).get("version"),
            "source_pinned_versions": {name: next((line.split("==", 1)[1].strip()
                                                   for line in requirements.splitlines()
                                                   if line.startswith(name + "==")), None)
                                       for name in ("numpy", "scipy", "tensorflow")},
            "arch_version_pinned": any(line.lower().startswith("arch==") for line in requirements.splitlines()),
            "cached_history_shallow": shallow, "available_commit_count": commit_count,
            "data_identical_to_retained_dataset": (authors / "data/raw/SP500_random.csv").read_bytes() == (root / "data/SP500_random.csv").read_bytes(),
            "recoverability": "The current source snapshot contains the historical sample, fitting procedure and non-matching aggregate GJR results, but neither the original full-precision calibration nor its initial state or checkpoints. The shallow local cache alone does not establish the contents of earlier upstream commits.",
        }
    return record


def scenarios(calibration):
    baseline = dict(calibration["selected_parameters"])
    observed = dict(baseline, initial_variance=calibration["authors_unscaled_fit"]["next_variance_forecast_from_observed_last_residual"])
    compatible = dict(baseline, initial_variance=7.75e-5)
    scaled_fit = calibration["scaled_conditioning_control"]
    scaled_p = scaled_fit["params_decimal_returns"]
    scaled = {"mu": scaled_p["mu"], "omega": scaled_p["omega"], "alpha": scaled_p["alpha[1]"],
              "gamma": scaled_p["gamma[1]"], "beta": scaled_p["beta[1]"],
              "initial_variance": scaled_fit["initial_fitted_variance_h0"]}
    rounded = dict(baseline, mu=.0006, alpha=.11, gamma=.20, beta=.78)
    literal = dict(rounded, omega=.0001)
    return [
        ("baseline_fitted_state", baseline, "Retained unscaled fit and terminal fitted variance."),
        ("observed_last_residual_forecast", observed, "Retained coefficients; known first-return variance is the one-step forecast from the final observed fitted residual."),
        ("illustrative_price_compatible_initial_variance", compatible, "Predefined price-bracketed variance 7.75e-5. Baseline and Panel B agents keep the retained fitted state; this variance is used only for the separately labelled sensitivity."),
        ("percentage_return_conditioning_fit_terminal_state", scaled, "Independent conditioning control obtained by fitting 100 times the returns and converting its fitted parameters back to decimal units."),
        ("printed_coefficients_hybrid_retained_intercept_and_state", rounded, "Hybrid sensitivity: printed mu=0.06%, alpha=0.11, gamma=0.20, beta=0.78, with the retained daily intercept and initial variance. Not the original calibration."),
        ("literal_printed_intercept_daily_variance_interpretation", literal, "Interpret 0.01% literally as omega=0.0001 in daily decimal variance units, together with other printed coefficients and the retained initial variance. This is a hypothetical unit interpretation, not an asserted meaning of the paper."),
    ]


def reuse_existing(root, parameters, seed):
    audit_path = root / "stable_price_matching_audit.json"
    if not audit_path.is_file():
        return None
    audit = read(audit_path)
    if audit["rates"] != {"r": .0167, "q": .0165} or audit["T_days"] != 63 or audit["S0"] != 100 or audit["K"] != 100:
        return None
    for row in audit["scenarios"]:
        expected = dict(row.get("parameters", {}), initial_variance=row.get("first_return_conditional_variance"))
        if expected == parameters and row.get("seed") == seed and row.get("paths") == 2000000:
            return {key: value for key, value in row.items() if key not in
                    ("parameters", "first_return_conditional_variance", "label", "price_3p16_in_ci95")}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--authors-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "calibration_price_diagnostics.json")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    authors = args.authors_root.expanduser().resolve() if args.authors_root else None
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration = read(root / "calibration.json")
    replication = load_module(root)
    result = {
        "author": __author__, "status": "in_progress",
        "historical_audit": historical_audit(root, authors, calibration),
        "pricing_method": "62 simulated Q periods plus exact final one-period conditional Gaussian put; put-call parity; 1,000,000 antithetic pairs per seed",
        "financial_inputs": {"S0": 100, "K": 100, "T_days": 63, "r": .0167, "q": .0165},
        "paths_per_seed": 2000000, "seeds": [46, 47],
        "interval_note": "Normal intervals are asymptotic intervals based on independent bounded antithetic-pair means, not exact finite-sample guarantees.",
        "selection_note": "Scenarios are fixed in advance and not selected by hedging losses. No policy is trained. The reused price-compatible variance is a previously bracketed illustrative state, not a recovered original input.",
        "scenarios": [],
    }
    write(output, result)
    for label, parameters, description in scenarios(calibration):
        record = {"label": label, "description": description, "parameters": parameters, "estimates": []}
        result["scenarios"].append(record)
        for seed in (46, 47):
            estimate = reuse_existing(root, parameters, seed)
            reused = estimate is not None
            if estimate is None:
                cfg = replication.Config(price_seed=seed)
                estimate = replication.risk_neutral_price(parameters, cfg, n=2000000)
            if (estimate["bound_stats"]["violations"] != 0 or estimate["paths_dropped"] != 0
                    or estimate["variance_clipped"] or not estimate["final_period_integrated_analytically"]):
                raise AssertionError("The bounded estimator checks failed")
            estimate["seed"] = seed
            estimate["reused_existing_calculation"] = reused
            estimate["ci95"] = [estimate["price"] - 1.96 * estimate["standard_error"],
                                estimate["price"] + 1.96 * estimate["standard_error"]]
            estimate["price_3p16_in_ci95"] = bool(estimate["ci95"][0] <= 3.16 <= estimate["ci95"][1])
            record["estimates"].append(estimate)
            write(output, result)
            print(f"{label}, seed {seed}: {estimate['price']:.9f} +/- {estimate['standard_error']:.9f} SE; reused={reused}", flush=True)
    result["status"] = "complete"
    write(output, result)
    print("Calibration and initial-state diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
