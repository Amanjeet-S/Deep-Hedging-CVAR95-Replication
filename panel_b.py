"""Reproduce Figure 1 Panel B on a common physical-measure test sample.

Examples, run from a reproduction directory:
    python panel_b.py train --confidence 85 --folder panel_b_runs
    python panel_b.py train --confidence 90 --folder panel_b_runs
    python panel_b.py evaluate --folder panel_b_runs

Use --root when the existing replication.py and run95 are elsewhere.
The 95% checkpoint and its recorded outputs are read without modification.
"""
from __future__ import annotations

__author__ = "Amanjeet Singh"

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import torch


PUBLISHED = {
    85: (1.979, -0.004, 1.280, -0.031),
    90: (2.412, -0.115, 1.433, -0.126),
    95: (3.481, -0.081, 1.810, -0.254),
}
COLOURS = {85: "#1f77b4", 90: "#17becf", 95: "#279e68"}
MODEL_PRICE = 2.7700684349135973
NAMES = (
    "Deep hedging CVaR loss",
    "Deep minus delta CVaR loss",
    "Difference strategy CVaR loss",
    "Difference strategy mean P&L",
)
UNCERTAINTY = (
    "Empirical plug-in intervals condition on fixed selected policies, retained "
    "parameters and the numerical delta benchmark. They exclude calibration, "
    "training and quadrature uncertainty. Required hedge-loss tail moments and "
    "normal population coverage have not been established."
)


def _module(root):
    filename = root / "replication.py"
    if not filename.is_file():
        raise FileNotFoundError("--root must contain replication.py")
    specification = importlib.util.spec_from_file_location("replication", filename)
    module = importlib.util.module_from_spec(specification)
    sys.modules["replication"] = module
    specification.loader.exec_module(module)
    return module


def _checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path.name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "config", "parameters", "epoch", "validation_cvar"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Incomplete checkpoint: {path.name}")
    return checkpoint


def _check_config(checkpoint, reference, confidence):
    current = checkpoint["config"]
    baseline = reference["config"]
    if set(current) != set(baseline):
        raise ValueError("Checkpoint configurations have different fields")
    for name, value in baseline.items():
        expected = confidence if name == "confidence" else value
        if current[name] != expected:
            raise ValueError(f"Checkpoint configuration differs in {name}")
    if checkpoint["parameters"] != reference["parameters"]:
        raise ValueError("All coefficients and the initial variance must agree exactly")
    if current["initial_capital"] != 3.16:
        raise ValueError("All checkpoints must use initial capital 3.16")


def _confidence(value):
    confidence = float(value)
    if confidence > 1:
        confidence /= 100
    if confidence not in (.85, .90):
        raise argparse.ArgumentTypeError("training confidence must be 85 or 90")
    return confidence


def train(args, replication):
    tag = int(round(args.confidence * 100))
    reference = _checkpoint(args.root / "run95" / "agent95.pt")
    configuration = replication.Config(confidence=args.confidence)
    candidate = {"config": vars(configuration), "parameters": reference["parameters"]}
    _check_config(candidate, reference, args.confidence)
    destination = args.folder / f"run{tag}"
    destination.mkdir(parents=True, exist_ok=True)
    temporary_name = destination / "agent95.pt"
    final_name = destination / f"agent{tag}.pt"
    if final_name.exists() or temporary_name.exists():
        raise FileExistsError("Training destination already contains a checkpoint")
    # The existing trainer uses agent95.pt as its generic checkpoint filename.
    # Rename only the completed local checkpoint; the reference run95 is untouched.
    replication.train(reference["parameters"], configuration, destination,
                      device=args.device, threads=1)
    history = json.loads((destination / "training_history.json").read_text())
    if len(history) != configuration.epochs or history[-1]["epoch"] != configuration.epochs:
        raise RuntimeError("Training has not completed all configured epochs")
    produced = _checkpoint(temporary_name)
    _check_config(produced, reference, args.confidence)
    produced["author"] = __author__
    torch.save(produced, temporary_name)
    temporary_name.replace(final_name)
    print(f"Completed {tag}% training; selected epoch {produced['epoch']}", flush=True)


def _evaluate_policy(replication, checkpoint, features, gains, capital=None):
    cfg = replication.Config(**checkpoint["config"])
    initial = cfg.initial_capital if capital is None else float(capital)
    model = torch.jit.script(replication.Hedger(cfg))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    n, days = gains.shape
    wealth = np.empty(n, dtype=np.float64)
    model_wealth = np.empty(n, dtype=np.float64)
    minimum_cash_model = math.inf
    minimum_cash_accumulation = math.inf
    minimum_cash_frozen_price = math.inf
    frozen_price_violating_paths = 0
    growth = math.exp(cfg.r / 252)
    with torch.no_grad():
        for start in range(0, n, cfg.batch_size):
            end = min(start + cfg.batch_size, n)
            terminal, positions, cash = model.positions(
                torch.from_numpy(features[start:end]),
                torch.from_numpy(gains[start:end]), initial)
            positions = positions.numpy().astype(np.float64)
            value = np.full(end - start, initial, dtype=np.float64)
            frozen_violations = np.zeros(end - start, dtype=bool)
            for t in range(days):
                stock = np.exp(features[start:end, t, 0].astype(np.float64))
                current_cash = value - positions[:, t] * stock
                minimum_cash_accumulation = min(
                    minimum_cash_accumulation,
                    float(np.min(current_cash)))
                # This changes the initial cash while keeping the returned
                # position path fixed. It does not reevaluate the feedback policy.
                frozen_price_cash = current_cash - (initial - MODEL_PRICE) * growth ** t
                minimum_cash_frozen_price = min(minimum_cash_frozen_price,
                                                float(frozen_price_cash.min()))
                frozen_violations |= frozen_price_cash < -cfg.borrowing_limit - .001
                value = growth * value + positions[:, t] * gains[start:end, t].astype(np.float64)
            wealth[start:end] = value
            model_wealth[start:end] = terminal.numpy().astype(np.float64)
            minimum_cash_model = min(minimum_cash_model, float(cash.min()))
            frozen_price_violating_paths += int(frozen_violations.sum())
    if not np.isfinite(wealth).all():
        raise FloatingPointError("Nonfinite policy wealth")
    checks = {
        "initial_capital": initial,
        "minimum_cash_float64": minimum_cash_accumulation,
        "minimum_cash_policy": minimum_cash_model,
        "torch_vs_float64_wealth_max_error": float(np.max(np.abs(wealth - model_wealth))),
        "frozen_positions_model_price_minimum_cash": minimum_cash_frozen_price,
        "frozen_positions_model_price_borrowing_violating_paths": frozen_price_violating_paths,
        "borrowing_tolerance": .001,
    }
    if minimum_cash_accumulation < -cfg.borrowing_limit - .001:
        raise AssertionError("Policy borrowing constraint exceeds its numerical tolerance")
    return wealth, checks


def _metrics(replication, deep_loss, delta_loss, overlay, confidence, published):
    n = len(deep_loss)
    se = lambda values: float(np.std(values, ddof=1) / np.sqrt(n))
    deep_if = replication.cvar_influence(deep_loss, confidence)
    delta_if = replication.cvar_influence(delta_loss, confidence)
    estimates = (
        replication.cvar(deep_loss, confidence),
        replication.cvar(deep_loss, confidence) - replication.cvar(delta_loss, confidence),
        replication.cvar(-overlay, confidence),
        float(overlay.mean()),
    )
    errors = (
        se(deep_if), se(deep_if - delta_if),
        se(replication.cvar_influence(-overlay, confidence)), se(overlay),
    )
    return [
        {"name": name, "published": target, "estimate": estimate,
         "standard_error": error, "ci_low": estimate - 1.96 * error,
         "ci_high": estimate + 1.96 * error, "difference": estimate - target}
        for name, target, estimate, error in zip(NAMES, published, estimates, errors)
    ]


def _capital_control(replication, cfg, loss, delta_loss, wealth, delta_wealth, overlay, checks):
    terminal_growth = math.exp(cfg.r * cfg.maturity / 252)
    shift = terminal_growth * (cfg.initial_capital - MODEL_PRICE)
    shifted_loss = loss + shift
    shifted_delta_loss = delta_loss + shift
    shifted_overlay = (wealth - shift) - (delta_wealth - shift)
    base_gap = replication.cvar(loss, cfg.confidence) - replication.cvar(delta_loss, cfg.confidence)
    shifted_gap = (replication.cvar(shifted_loss, cfg.confidence)
                   - replication.cvar(shifted_delta_loss, cfg.confidence))
    overlay_error = float(np.max(np.abs(shifted_overlay - overlay)))
    if abs(shifted_gap - base_gap) > 1e-10 or overlay_error > 1e-10:
        raise AssertionError("Fixed-position common-capital cancellation failed")
    return {
        "label": "Accounting diagnostic: initial capital changed to model price with both position paths frozen",
        "original_initial_capital": cfg.initial_capital,
        "control_initial_capital": MODEL_PRICE,
        "terminal_loss_increase": shift,
        "deep_cvar": replication.cvar(shifted_loss, cfg.confidence),
        "delta_cvar": replication.cvar(shifted_delta_loss, cfg.confidence),
        "cvar_gap": shifted_gap,
        "gap_change": shifted_gap - base_gap,
        "overlay_identity_max_error": overlay_error,
        "overlay_cvar": replication.cvar(-shifted_overlay, cfg.confidence),
        "overlay_mean": float(shifted_overlay.mean()),
        "minimum_cash_deep_hedge": checks["frozen_positions_model_price_minimum_cash"],
        "borrowing_violating_deep_paths": checks["frozen_positions_model_price_borrowing_violating_paths"],
        "borrowing_tolerance": checks["borrowing_tolerance"],
        "deep_hedge_admissible_with_tolerance": checks["frozen_positions_model_price_borrowing_violating_paths"] == 0,
        "interpretation": "This is a cash-translation accounting diagnostic, not a policy rerun or retraining. Frozen positions can violate the borrowing limit after reducing initial cash; admissibility is not presumed.",
    }


def _draw(folder, samples, bins, full_range=False):
    os.environ.setdefault("MPLCONFIGDIR", str(folder / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    for tag in (85, 90, 95):
        pnl = samples[f"overlay_pnl{tag}"]
        counts, _ = np.histogram(pnl, bins=bins)
        ax.stairs(counts, bins, color=COLOURS[tag], fill=True, alpha=.12)
        ax.stairs(counts, bins, color=COLOURS[tag], linewidth=1.7, label=f"CVaR {tag}% agent")
        ax.axvline(pnl.mean(), color=COLOURS[tag], linestyle=":", linewidth=1)
    ax.axvline(0, color="#17384f", linewidth=.9)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("Terminal P&L of the deep-minus-delta strategy")
    ax.set_ylabel("Frequency (paths)")
    ax.set_title("Figure 1, Panel B: independent reproduction", fontsize=13)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    if full_range:
        ax.set_yscale("log")
        filename = "figure1_panel_b_full_range"
    else:
        filename = "figure1_panel_b"
    footer = "Common 100,000-path test sample; dotted lines mark means"
    if full_range:
        footer += "\nAll outcomes shown; frequencies use a logarithmic scale"
    else:
        outside = [int(np.count_nonzero((samples[f'overlay_pnl{tag}'] < -5)
                                      | (samples[f'overlay_pnl{tag}'] > 5)))
                   for tag in (85, 90, 95)]
        footer += f"\nBins of width 0.1; outside displayed range: 85% = {outside[0]}, 90% = {outside[1]}, 95% = {outside[2]}"
    fig.text(.5, .02, footer, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .11, 1, .94))
    fig.savefig(folder / f"{filename}.png", dpi=220,
                metadata={"Author": __author__, "Title": "Independent Figure 1 Panel B"})
    fig.savefig(folder / f"{filename}.pdf",
                metadata={"Author": __author__, "Title": "Independent Figure 1 Panel B"})
    plt.close(fig)


def evaluate(args, replication):
    reference = _checkpoint(args.root / "run95" / "agent95.pt")
    checkpoints = {
        85: _checkpoint(args.folder / "run85" / "agent85.pt"),
        90: _checkpoint(args.folder / "run90" / "agent90.pt"),
        95: reference,
    }
    for tag, checkpoint in checkpoints.items():
        _check_config(checkpoint, reference, tag / 100)
        history_path = (args.root / "run95" if tag == 95 else args.folder / f"run{tag}") / "training_history.json"
        history = json.loads(history_path.read_text())
        if len(history) != checkpoint["config"]["epochs"] or history[-1]["epoch"] != checkpoint["config"]["epochs"]:
            raise RuntimeError(f"The {tag}% training run has not completed all epochs")
    cfg = replication.Config(**reference["config"])
    if cfg.test_paths != 100000 or cfg.test_seed != 45:
        raise ValueError("The shared test sample must be 100,000 paths with seed 45")
    replication.configure_torch(1)
    features, gains, payoff32 = replication.physical_paths(
        cfg.test_paths, reference["parameters"], cfg, cfg.test_seed)
    payoff = payoff32.astype(np.float64)
    with np.load(args.root / "run95" / "terminal_results.npz", allow_pickle=False) as saved:
        delta_loss = saved["delta_loss"].copy()
        original_deep95 = saved["deep_loss"].copy()
        original_overlay95 = saved["overlay_pnl"].copy()
    if any(len(values) != cfg.test_paths for values in (delta_loss, original_deep95, original_overlay95)):
        raise ValueError("Recorded 95% outcomes have the wrong sample size")
    saved_results = json.loads((args.root / "run95" / "results.json").read_text())
    if saved_results["config"] != reference["config"] or saved_results["parameters"] != reference["parameters"]:
        raise ValueError("Recorded test configuration does not match the reference checkpoint")
    delta_wealth = payoff - delta_loss
    samples = {"payoff": payoff, "delta_loss": delta_loss, "delta_wealth": delta_wealth}
    report = {"author": __author__, "test_paths": cfg.test_paths, "test_seed": cfg.test_seed,
              "common_market_paths": True, "parameters": reference["parameters"],
              "agents": {}, "uncertainty_note": UNCERTAINTY,
              "histogram": {"range": [-5, 5], "bins": 100, "width": .1,
                            "note": "Common bins defined for this independent reproduction; the original published bin settings are not disclosed in the inspected implementation."}}
    capital = {"author": __author__, "checkpoint_initial_capital": {}, "model_price": MODEL_PRICE,
               "terminal_cash_growth": math.exp(cfg.r * cfg.maturity / 252),
               "fixed_position_controls": {},
               "wealth_feedback_note": "Changing initial wealth while rerunning the policy can change its wealth input and borrowing cap. Cash-translation invariance applies only with positions fixed."}
    bins = np.linspace(-5, 5, 101)
    histogram = []
    for tag in (95, 85, 90):
        checkpoint = checkpoints[tag]
        own_cfg = replication.Config(**checkpoint["config"])
        wealth, checks = _evaluate_policy(replication, checkpoint, features, gains)
        loss = payoff - wealth
        overlay = wealth - delta_wealth
        if not (np.isfinite(loss).all() and np.isfinite(overlay).all()):
            raise FloatingPointError("Nonfinite terminal loss or difference-strategy profit")
        if tag == 95:
            loss_error = float(np.max(np.abs(loss - original_deep95)))
            overlay_error = float(np.max(np.abs(overlay - original_overlay95)))
            if loss_error > 1e-8 or overlay_error > 1e-8:
                raise AssertionError("Recomputed 95% outcomes differ from the preserved original run")
            report["reference95_checks"] = {"deep_loss_max_error": loss_error,
                                             "overlay_max_error": overlay_error}
        metrics = _metrics(replication, loss, delta_loss, overlay, own_cfg.confidence, PUBLISHED[tag])
        if tag == 95:
            for current, recorded in zip(metrics, saved_results["metrics"]):
                if abs(current["estimate"] - recorded["estimate"]) > 1e-8:
                    raise AssertionError("The preserved 95% metric estimate was not recovered")
        counts, _ = np.histogram(overlay, bins=bins)
        histogram.append((tag, counts))
        report["agents"][str(tag)] = {
            "confidence": own_cfg.confidence, "config": checkpoint["config"],
            "best_epoch": checkpoint["epoch"], "best_validation_cvar": checkpoint["validation_cvar"],
            "metrics": metrics, "delta_cvar": replication.cvar(delta_loss, own_cfg.confidence),
            "overlay_mean": float(overlay.mean()),
            "overlay_range": [float(overlay.min()), float(overlay.max())],
            "outside_central_range": int(np.count_nonzero((overlay < -5) | (overlay > 5))),
            "below_central_range": int(np.count_nonzero(overlay < -5)),
            "above_central_range": int(np.count_nonzero(overlay > 5)),
            "central_histogram_count": int(counts.sum()),
            "statistical_arbitrage_criterion_met_in_sample": metrics[2]["estimate"] < 0,
            "checks": checks,
        }
        samples[f"deep_loss{tag}"] = loss
        samples[f"deep_wealth{tag}"] = wealth
        samples[f"overlay_pnl{tag}"] = overlay
        capital["checkpoint_initial_capital"][str(tag)] = own_cfg.initial_capital
        capital["fixed_position_controls"][str(tag)] = _capital_control(
            replication, own_cfg, loss, delta_loss, wealth, delta_wealth, overlay, checks)
        print(f"Evaluated {tag}% agent; mean difference-strategy profit {overlay.mean():.6f}", flush=True)

    if args.capital_control:
        wealth, checks = _evaluate_policy(replication, reference, features, gains, MODEL_PRICE)
        shift = capital["terminal_cash_growth"] * (cfg.initial_capital - MODEL_PRICE)
        control_delta_wealth = delta_wealth - shift
        control_delta_loss = delta_loss + shift
        control_loss = payoff - wealth
        control_overlay = wealth - control_delta_wealth
        capital["frozen_weights95_policy_rerun"] = {
            "label": "95% weights fixed, policy rerun at model-price capital; not retraining",
            "runtime_initial_capital": MODEL_PRICE, "checkpoint_initial_capital": cfg.initial_capital,
            "metrics": _metrics(replication, control_loss, control_delta_loss,
                                control_overlay, cfg.confidence, PUBLISHED[95]),
            "checks": checks,
            "overlay_change_max": float(np.max(np.abs(control_overlay - samples["overlay_pnl95"]))),
        }
        samples["capital_control95_deep_loss"] = control_loss
        samples["capital_control95_overlay_pnl"] = control_overlay
        samples["capital_control_delta_loss"] = control_delta_loss

    report["agents"] = {str(tag): report["agents"][str(tag)] for tag in (85, 90, 95)}
    args.folder.mkdir(parents=True, exist_ok=True)
    (args.folder / "panel_b_results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.folder / "capital_audit.json").write_text(json.dumps(capital, indent=2, allow_nan=False))
    np.savez_compressed(args.folder / "panel_b_terminal_results.npz", **samples)
    counts = dict(histogram)
    np.savetxt(args.folder / "panel_b_histogram.csv",
               np.column_stack((bins[:-1], bins[1:], counts[85], counts[90], counts[95])),
               delimiter=",", header="bin_left,bin_right,count85,count90,count95", comments="")
    _draw(args.folder, samples, bins)
    if args.full_range:
        all_pnl = np.concatenate([samples[f"overlay_pnl{tag}"] for tag in (85, 90, 95)])
        _draw(args.folder, samples, np.linspace(all_pnl.min(), all_pnl.max(), 101), full_range=True)
    print("Saved Panel B results, terminal outcomes, common histogram and capital audit", flush=True)


def self_test(replication):
    rng = np.random.RandomState(20261007)
    n, days, r, q = 2000, 4, .0167, .0165
    growth = math.exp(r / 252)
    stock = np.full(n, 100.0)
    wealth_a = np.full(n, 3.16)
    wealth_b = wealth_a.copy()
    shifted_a = np.full(n, 3.41)
    shifted_b = shifted_a.copy()
    overlay = np.zeros(n)
    for t in range(days):
        next_stock = stock * np.exp(.0002 + .01 * rng.normal(size=n))
        gain = math.exp(q / 252) * next_stock - growth * stock
        delta_a = .6 + .02 * t
        delta_b = .45 - .01 * t
        cash_reference = growth * (wealth_a - delta_a * stock) + math.exp(q / 252) * delta_a * next_stock
        wealth_a = growth * wealth_a + delta_a * gain
        if np.max(np.abs(wealth_a - cash_reference)) > 1e-12:
            raise AssertionError("Separate cash and stock financing disagree")
        wealth_b = growth * wealth_b + delta_b * gain
        shifted_a = growth * shifted_a + delta_a * gain
        shifted_b = growth * shifted_b + delta_b * gain
        overlay = growth * overlay + (delta_a - delta_b) * gain
        stock = next_stock
    shift = .25 * growth ** days
    payoff = np.maximum(stock - 100, 0)
    max_overlay_error = float(np.max(np.abs(overlay - (wealth_a - wealth_b))))
    max_translation_error = float(np.max(np.abs(shifted_a - wealth_a - shift)))
    if max_overlay_error > 1e-12 or max_translation_error > 1e-12:
        raise AssertionError("Wealth translation or zero-capital identity failed")
    for confidence in (.85, .90, .95):
        loss_a, loss_b = payoff - wealth_a, payoff - wealth_b
        base_gap = replication.cvar(loss_a, confidence) - replication.cvar(loss_b, confidence)
        shifted_gap = (replication.cvar(payoff - shifted_a, confidence)
                       - replication.cvar(payoff - shifted_b, confidence))
        if abs(base_gap - shifted_gap) > 1e-12:
            raise AssertionError("CVaR gap cash translation failed")
    print(json.dumps({"passed": True, "overlay_identity_max_error": max_overlay_error,
                      "capital_translation_max_error": max_translation_error,
                      "tested_confidences": [.85, .90, .95]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "self-test"):
        sub = subcommands.add_parser(command)
        sub.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                         help="directory containing replication.py and preserved run95")
        if command != "self-test":
            sub.add_argument("--folder", type=Path, required=True,
                             help="directory containing run85/run90 and new Panel B outputs")
        if command == "train":
            sub.add_argument("--confidence", type=_confidence, required=True,
                             help="85 or 90 (also accepts 0.85 or 0.90)")
            sub.add_argument("--device", choices=("cpu", "mps"), default="cpu")
        if command == "evaluate":
            sub.add_argument("--capital-control", action="store_true",
                             help="also rerun fixed 95%% weights at model-price initial capital")
            sub.add_argument("--full-range", action="store_true")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    if hasattr(args, "folder"):
        args.folder = args.folder.expanduser().resolve()
    replication = _module(args.root)
    if args.command == "train":
        train(args, replication)
    elif args.command == "evaluate":
        evaluate(args, replication)
    else:
        self_test(replication)


if __name__ == "__main__":
    main()
