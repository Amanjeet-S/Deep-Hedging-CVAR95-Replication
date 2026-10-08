"""Evaluate predefined calibration sensitivities without further policy training.

Paper: François, Gauthier, Godin and Pérez Mendoza (2025),
Is the difference between deep hedging and delta hedging a statistical arbitrage?,
Finance Research Letters 73, 106590, doi:10.1016/j.frl.2024.106590.
Conditional deltas and portfolio accounting use the existing replication modules.
"""
from __future__ import annotations

__author__ = "Amanjeet Singh"

import argparse
import copy
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def load_modules(root):
    sys.path.insert(0, str(root))
    from replication import Config, Hedger, physical_paths, cvar, cvar_influence, configure_torch
    from delta_reference import DeltaGrid, GJRParameters, GridConfig
    return Config, Hedger, physical_paths, cvar, cvar_influence, configure_torch, DeltaGrid, GJRParameters, GridConfig


def prepare_inputs(root, folder):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    baseline = json.loads((root / 'calibration.json').read_text())
    compatible = copy.deepcopy(baseline)
    compatible['author'] = __author__
    compatible['scenario_label'] = 'initial-variance sensitivity, not recovered original calibration'
    compatible['selected_parameters']['initial_variance'] = 7.75e-5
    control = baseline['scaled_conditioning_control']
    p = control['params_decimal_returns']
    scaled = copy.deepcopy(baseline)
    scaled['author'] = __author__
    scaled['scenario_label'] = 'unit-scaled likelihood sensitivity, not recovered original calibration'
    scaled['selected_parameters'] = dict(mu=p['mu'], omega=p['omega'], alpha=p['alpha[1]'],
        gamma=p['gamma[1]'], beta=p['beta[1]'], initial_variance=control['initial_fitted_variance_h0'])
    for name, data in [('price_compatible', compatible), ('scaled_fit', scaled)]:
        data['selection_note'] = 'Prespecified sensitivity inputs. All financial and training settings retain the 95% baseline; initial hedging capital remains 3.16.'
        destination = folder / name / 'calibration.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            saved = json.loads(destination.read_text())
            assert saved['selected_parameters'] == data['selected_parameters'], 'Existing scenario inputs differ'
        else:
            destination.write_text(json.dumps(data, indent=2))
    prices = root / 'calibration_sensitivity/calibration_price_diagnostics.json'
    if prices.is_file() and not (folder / prices.name).exists():
        (folder / prices.name).write_bytes(prices.read_bytes())
    print('Prepared the two predefined sensitivity calibrations.', flush=True)


def sampling_record(estimate, influence):
    error = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    return {'estimate': float(estimate), 'standard_error': error,
            'ci_low': float(estimate - 1.96 * error), 'ci_high': float(estimate + 1.96 * error)}


def read_checkpoint(path, expected_config, expected_parameters=None, history_path=None):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    assert ck['config'] == asdict(expected_config), 'Checkpoint configuration differs from the common 95% configuration'
    if expected_parameters is not None:
        assert ck['parameters'] == expected_parameters, 'Checkpoint parameters differ from the scenario calibration'
    if history_path is not None:
        history = json.loads(Path(history_path).read_text())
        assert [row['epoch'] for row in history] == list(range(1, 51)), 'All 50 epochs must be present'
        best = min(history, key=lambda row: row['validation_cvar'])
        assert ck['epoch'] == best['epoch']
        assert ck['validation_cvar'] == best['validation_cvar']
    assert all(bool(torch.isfinite(value).all()) for value in ck['state_dict'].values())
    return ck


def load_or_build_grid(params, cfg, modules, cache=None, label=''):
    _, _, _, _, _, _, DeltaGrid, GJRParameters, GridConfig = modules
    gp = GJRParameters(**{key: params[key] for key in ['mu', 'omega', 'alpha', 'beta', 'gamma']}, r=cfg.r, q=cfg.q)
    fine = GridConfig(x_min=-1.5, x_max=1.5, nx=4801, logh_min=-12, logh_max=-1, nh=221, quadrature_nodes=48)
    if cache is not None and Path(cache).is_file():
        grid = DeltaGrid.load(cache)
        assert asdict(grid.params) == asdict(gp), 'Cached delta coefficients differ from this market'
        assert asdict(grid.config) == asdict(fine), 'Cached delta grid resolution differs'
        assert grid.maturity_days == cfg.maturity
        print(f'{label}: loaded matching fine delta cache', flush=True)
    else:
        print(f'{label}: building the market-specific fine delta grid', flush=True)
        grid = DeltaGrid.build(gp, cfg.maturity, fine,
            progress=lambda day, secs: print(f'{label} fine grid {day}/{cfg.maturity}: {secs:.1f}s', flush=True))
    grid.enable_tail_refinement(threshold=.01,
        progress=lambda day, secs: print(f'{label} wide grid {day}/{cfg.maturity}: {secs:.1f}s', flush=True))
    return grid


def evaluate_market(params, cfg, checkpoints, grid, modules, label):
    _, Hedger, physical_paths, cvar, cvar_influence, _, _, _, _ = modules
    initial_delta = {'spot': cfg.spot, 'initial_variance': params['initial_variance'],
                     'remaining_days': cfg.maturity,
                     'estimate': float(grid.delta(cfg.maturity, cfg.spot, params['initial_variance'], cfg.strike))}
    print(f'{label}: generating {cfg.test_paths:,} physical test paths with seed {cfg.test_seed}', flush=True)
    features, gains, payoff = physical_paths(cfg.test_paths, params, cfg, cfg.test_seed)
    assert np.isfinite(features).all() and np.isfinite(gains).all() and np.isfinite(payoff).all()
    n = cfg.test_paths
    growth = np.exp(cfg.r / 252)
    vdelta = np.full(n, cfg.initial_capital, dtype=np.float64)
    deltas = np.empty((n, cfg.maturity), dtype=np.float64)
    for day in range(cfg.maturity):
        spot = np.exp(features[:, day, 0].astype(np.float64))
        variance = features[:, day, 2].astype(np.float64)**2 / 252
        deltas[:, day] = grid.delta(cfg.maturity - day, spot, variance, cfg.strike)
        vdelta = growth * vdelta + deltas[:, day] * gains[:, day].astype(np.float64)
    assert np.isfinite(deltas).all() and np.isfinite(vdelta).all()
    delta_loss = payoff.astype(np.float64) - vdelta
    delta_influence = cvar_influence(delta_loss, cfg.confidence)
    delta_record = sampling_record(cvar(delta_loss, cfg.confidence), delta_influence)
    rows = {}
    terminals = {'delta_loss': delta_loss}
    for policy_label, ck in checkpoints.items():
        print(f'{label}: evaluating {policy_label}', flush=True)
        model = torch.jit.script(Hedger(cfg))
        model.load_state_dict(ck['state_dict'])
        model.eval()
        stock_positions = np.empty((n, cfg.maturity), dtype=np.float32)
        torch_wealth = np.empty(n, dtype=np.float32)
        torch_min_cash = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, cfg.batch_size):
                stop = min(start + cfg.batch_size, n)
                wealth, positions, minimum_cash = model.positions(
                    torch.from_numpy(features[start:stop]), torch.from_numpy(gains[start:stop]), cfg.initial_capital)
                torch_wealth[start:stop] = wealth.numpy()
                stock_positions[start:stop] = positions.numpy()
                torch_min_cash[start:stop] = minimum_cash.numpy()
        assert np.isfinite(stock_positions).all() and np.isfinite(torch_wealth).all() and np.isfinite(torch_min_cash).all()
        vdh = np.full(n, cfg.initial_capital, dtype=np.float64)
        overlay = np.zeros(n, dtype=np.float64)
        min_cash = np.full(n, np.inf, dtype=np.float64)
        for day in range(cfg.maturity):
            spot = np.exp(features[:, day, 0].astype(np.float64))
            position = stock_positions[:, day].astype(np.float64)
            gain = gains[:, day].astype(np.float64)
            min_cash = np.minimum(min_cash, vdh - position * spot)
            vdh = growth * vdh + position * gain
            overlay = growth * overlay + (position - deltas[:, day]) * gain
        deep_loss = payoff.astype(np.float64) - vdh
        deep_influence = cvar_influence(deep_loss, cfg.confidence)
        overlay_influence = cvar_influence(-overlay, cfg.confidence)
        estimates = [cvar(deep_loss, cfg.confidence), cvar(deep_loss, cfg.confidence) - delta_record['estimate'],
                     cvar(-overlay, cfg.confidence), float(overlay.mean())]
        influences = [deep_influence, deep_influence - delta_influence, overlay_influence, overlay]
        names = ['Deep hedging CVaR95 loss', 'Deep minus delta CVaR95 loss',
                 'Difference strategy CVaR95 loss', 'Difference strategy mean P&L']
        published = [3.481, -.081, 1.810, -.254]
        metrics = []
        for name, estimate, influence, target in zip(names, estimates, influences, published):
            metric = {'name': name, **sampling_record(estimate, influence), 'published': target,
                      'difference': float(estimate - target)}
            metrics.append(metric)
        checks = {
            'overlay_identity_max_error': float(np.max(np.abs(overlay - (vdh - vdelta)))),
            'hedging_loss_identity_max_error': float(np.max(np.abs(deep_loss - (delta_loss - overlay)))),
            'torch_vs_float64_wealth_max_error': float(np.max(np.abs(vdh - torch_wealth))),
            'minimum_cash_float64': float(min_cash.min()),
            'minimum_cash_torch': float(torch_min_cash.min()),
            'borrowing_limit_tolerance': .001,
            'finite_test_losses': bool(np.isfinite(deep_loss).all() and np.isfinite(delta_loss).all()),
            'finite_overlay_and_positions': bool(np.isfinite(overlay).all() and np.isfinite(stock_positions).all()),
            'training_test_distinct_seeds': cfg.training_seed != cfg.test_seed,
            'validation_test_distinct_seeds': cfg.validation_seed != cfg.test_seed,
        }
        assert checks['overlay_identity_max_error'] < 1e-8
        assert checks['hedging_loss_identity_max_error'] < 1e-8
        assert checks['minimum_cash_float64'] >= -cfg.borrowing_limit - .001
        assert checks['finite_test_losses'] and checks['finite_overlay_and_positions']
        row = {
            'author': __author__, 'policy': policy_label, 'metrics': metrics, 'config': asdict(cfg),
            'parameters': params, 'policy_training_parameters': ck['parameters'],
            'best_epoch': ck['epoch'], 'best_validation_cvar': ck['validation_cvar'],
            'delta_cvar': delta_record['estimate'], 'delta_standard_error': delta_record['standard_error'],
            'delta_confidence_interval': [delta_record['ci_low'], delta_record['ci_high']],
            'initial_conditional_delta': initial_delta,
            'checks': checks, 'overlay_quantiles': {str(q): float(np.quantile(overlay, q)) for q in [.001, .01, .05, .5, .95, .99, .999]},
            'overlay_range': [float(overlay.min()), float(overlay.max())],
            'outside_central_plot': int(np.count_nonzero((overlay < -5) | (overlay > 5))),
            'statistical_arbitrage_criterion_met': estimates[2] < 0,
            'conditional_uncertainty_note': 'Sampling intervals condition on one selected policy, the scenario parameters and its numerical delta benchmark. They exclude training-seed, parameter and grid uncertainty.',
        }
        rows[policy_label] = row
        terminals[f'{policy_label}_deep_loss'] = deep_loss
        terminals[f'{policy_label}_overlay_pnl'] = overlay
        print(f"{label} {policy_label}: {[round(value, 6) for value in estimates]}", flush=True)
        del model, stock_positions, torch_wealth, torch_min_cash
        gc.collect()
    diagnostic = grid.diagnostic_summary()
    diagnostic['evaluation_counter_scope'] = 'Cumulative across scenarios sharing this in-memory coefficient grid.'
    for row in rows.values():
        row['delta_grid'] = diagnostic
    return rows, terminals


def save_case(folder, case_name, label, params, cfg, checkpoints, grid, modules, baseline_loss, price_scenarios):
    case_folder = folder / case_name
    rows, terminals = evaluate_market(params, cfg, checkpoints, grid, modules, case_name)
    _, _, _, cvar, cvar_influence, _, _, _, _ = modules
    base_cvar = cvar(baseline_loss, cfg.confidence)
    fixed_loss = terminals['fixed_baseline_weights_deep_loss']
    fixed_cvar = cvar(fixed_loss, cfg.confidence)
    component_a = sampling_record(fixed_cvar - base_cvar,
        cvar_influence(fixed_loss, cfg.confidence) - cvar_influence(baseline_loss, cfg.confidence))
    decomposition = {'A_market_change_fixed_baseline_policy': component_a,
                     'definition': 'A=rho_new(theta_baseline)-rho_baseline(theta_baseline); B=rho_new(theta_new)-rho_new(theta_baseline).',
                     'interpretation': 'A controlled simulation comparison using common random numbers, not causal identification of the unpublished original calibration.'}
    if 'retrained' in rows:
        new_loss = terminals['retrained_deep_loss']
        new_cvar = cvar(new_loss, cfg.confidence)
        component_b = sampling_record(new_cvar - fixed_cvar,
            cvar_influence(new_loss, cfg.confidence) - cvar_influence(fixed_loss, cfg.confidence))
        total = sampling_record(new_cvar - base_cvar,
            cvar_influence(new_loss, cfg.confidence) - cvar_influence(baseline_loss, cfg.confidence))
        error = abs(component_a['estimate'] + component_b['estimate'] - total['estimate'])
        assert error < 1e-12
        decomposition.update({'B_policy_retraining_on_new_market': component_b,
                              'total_change': total, 'A_plus_B_identity_max_error': error})
    else:
        decomposition.update({'B_policy_retraining_on_new_market': None,
                              'total_change_fixed_policy': component_a, 'retraining_note': 'No new policy was trained for this scenario.'})
    for policy_name, row in rows.items():
        row['scenario_label'] = label
        row_folder = case_folder / ('run95' if policy_name == 'retrained' else 'fixed_baseline')
        row_folder.mkdir(parents=True, exist_ok=True)
        (row_folder / 'results.json').write_text(json.dumps(row, indent=2))
        np.savez_compressed(row_folder / 'terminal_results.npz',
            deep_loss=terminals[f'{policy_name}_deep_loss'], delta_loss=terminals['delta_loss'],
            overlay_pnl=terminals[f'{policy_name}_overlay_pnl'])
    np.savez_compressed(case_folder / 'sensitivity_terminal_results.npz', **terminals)
    case_result = {'name': case_name, 'scenario_label': label, 'parameters': params,
                   'option_price_diagnostics': price_scenarios.get(case_name),
                   'policies': list(rows.values()), 'decomposition': decomposition,
                   'terminal_results': f'{case_name}/sensitivity_terminal_results.npz'}
    (case_folder / 'sensitivity_results.json').write_text(json.dumps({'author': __author__, **case_result}, indent=2))
    return case_result


def run(root, folder, baseline_grid=None):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    modules = load_modules(root)
    Config, _, _, _, _, configure_torch, _, _, _ = modules
    cfg = Config()
    configure_torch(1)
    base_calibration = json.loads((root / 'calibration.json').read_text())
    base_params = base_calibration['selected_parameters']
    baseline_ck = read_checkpoint(root / 'run95/agent95.pt', cfg, base_params,
                                  root / 'run95/training_history.json')
    forecast_folder = folder / 'observed_forecast'
    forecast_folder.mkdir(parents=True, exist_ok=True)
    forecast_cal = copy.deepcopy(base_calibration)
    forecast_cal['author'] = __author__
    forecast_cal['scenario_label'] = 'observed-residual initial-variance sensitivity; fixed baseline policy only'
    forecast_cal['selected_parameters'] = copy.deepcopy(base_params)
    forecast_cal['selected_parameters']['initial_variance'] = base_calibration['authors_unscaled_fit']['next_variance_forecast_from_observed_last_residual']
    forecast_cal['selection_note'] = 'Only the known first-return variance is replaced by the observed-residual one-step forecast. No policy retraining or selection on hedging losses.'
    (forecast_folder / 'calibration.json').write_text(json.dumps(forecast_cal, indent=2))
    calibration_files = {name: json.loads((folder / name / 'calibration.json').read_text())
                         for name in ['observed_forecast', 'price_compatible', 'scaled_fit']}
    new_checkpoints = {}
    for name in ['price_compatible', 'scaled_fit']:
        new_checkpoints[name] = read_checkpoint(folder / name / 'run95/agent95.pt', cfg,
            calibration_files[name]['selected_parameters'], folder / name / 'run95/training_history.json')
    price_scenarios = {}
    price_path = folder / 'calibration_price_diagnostics.json'
    if price_path.is_file():
        price_report = json.loads(price_path.read_text())
        labels = {'observed_forecast': 'observed_last_residual_forecast',
                  'price_compatible': 'illustrative_price_compatible_initial_variance',
                  'scaled_fit': 'percentage_return_conditioning_fit_terminal_state'}
        for name, price_label in labels.items():
            source = next((row for row in price_report['scenarios'] if row['label'] == price_label), None)
            if source is not None:
                assert source['parameters'] == calibration_files[name]['selected_parameters']
                price_scenarios[name] = {'label': source['label'], 'estimates': source['estimates'],
                                         'description_at_price_audit': source['description']}
    grid = load_or_build_grid(base_params, cfg, modules, baseline_grid, 'baseline-coefficient cases')
    baseline_terminal = root / 'run95/terminal_results.npz'
    if baseline_terminal.is_file():
        with np.load(baseline_terminal) as arrays:
            baseline_loss = arrays['deep_loss'].copy()
        base_results = json.loads((root / 'run95/results.json').read_text())
        assert base_results['config'] == asdict(cfg) and base_results['parameters'] == base_params
        assert len(baseline_loss) == cfg.test_paths and np.isfinite(baseline_loss).all()
        assert abs(modules[3](baseline_loss, cfg.confidence) - base_results['metrics'][0]['estimate']) < 1e-10
    else:
        _, base_terminals = evaluate_market(base_params, cfg, {'fixed_baseline_weights': baseline_ck},
                                            grid, modules, 'baseline reference')
        baseline_loss = base_terminals['fixed_baseline_weights_deep_loss']
        del base_terminals
    cases = []
    for name in ['observed_forecast', 'price_compatible']:
        calibration = calibration_files[name]
        for key in ['mu', 'omega', 'alpha', 'beta', 'gamma']:
            assert calibration['selected_parameters'][key] == base_params[key]
        checkpoints = {'fixed_baseline_weights': baseline_ck}
        if name in new_checkpoints:
            checkpoints['retrained'] = new_checkpoints[name]
        cases.append(save_case(folder, name, calibration['scenario_label'], calibration['selected_parameters'],
            cfg, checkpoints, grid, modules, baseline_loss, price_scenarios))
        gc.collect()
    del grid
    gc.collect()
    print('Released baseline fine and wide delta arrays before scaled-fit grid construction', flush=True)
    scaled = calibration_files['scaled_fit']
    grid = load_or_build_grid(scaled['selected_parameters'], cfg, modules, label='scaled-fit case')
    cases.append(save_case(folder, 'scaled_fit', scaled['scenario_label'], scaled['selected_parameters'], cfg,
        {'fixed_baseline_weights': baseline_ck, 'retrained': new_checkpoints['scaled_fit']},
        grid, modules, baseline_loss, price_scenarios))
    del grid
    gc.collect()
    combined = {
        'author': __author__, 'status': 'complete', 'config': asdict(cfg),
        'baseline_parameters': base_params, 'baseline_deep_cvar': modules[3](baseline_loss, cfg.confidence),
        'policy_selection': 'Existing completed 50-epoch checkpoints selected by their independent validation samples. No training or parameter selection in this evaluator.',
        'capital_note': 'All policies start with 3.16, including fixed-baseline-weight evaluations on each new market.',
        'uncertainty_note': 'Intervals are empirical plug-in estimates conditional on the selected policies and models. Paired decomposition estimates use common market innovations from seed 45 and exclude calibration, repeat-training and delta-grid uncertainty. Required influence-function moments and normal population coverage have not been established.',
        'interpretation': 'Predefined calibration and initial-state sensitivities, not recovered original calibration or causal identification of differences from the published table.',
        'cases': cases,
    }
    (folder / 'sensitivity_results.json').write_text(json.dumps(combined, indent=2))
    print(json.dumps({'status': 'complete', 'baseline_deep_cvar': combined['baseline_deep_cvar'],
        'cases': [{'name': case['name'], 'policies': [{'policy': row['policy'],
            'metrics': [metric['estimate'] for metric in row['metrics']], 'delta_cvar': row['delta_cvar']}
            for row in case['policies']], 'decomposition': case['decomposition']} for case in cases]}, indent=2), flush=True)
    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Folder containing the unchanged replication modules and baseline run95')
    parser.add_argument('--folder', type=Path, required=True,
                        help='Folder containing price_compatible and scaled_fit calibration/run95 inputs')
    parser.add_argument('--baseline-grid', type=Path,
                        help='Optional matching fine-grid cache; absent caches are rebuilt')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Write the two predefined calibration inputs before retraining')
    args = parser.parse_args()
    if args.prepare_only:
        prepare_inputs(args.root, args.folder)
    else:
        run(args.root, args.folder, args.baseline_grid)
