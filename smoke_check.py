"""Check the supplied research snapshot without training or resimulation.

Checks consistency across independently saved research artefacts.
Research implementation: Amanjeet Singh (https://github.com/Amanjeet-S).
Run from any directory: python smoke_check.py [--root PATH_TO_REPOSITORY]
"""
from __future__ import annotations

__author__ = "Amanjeet Singh"

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def empirical_cvar(values: np.ndarray, confidence: float) -> float:
    """Mean of the upper 5,000 observations in this 100,000-path snapshot."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    start = int(np.floor(confidence * len(ordered)))
    return float(np.mean(ordered[start:]))


def check_snapshot(root: Path) -> dict:
    root = root.resolve()
    results = json.loads((root / 'run95/results.json').read_text())
    calibration = json.loads((root / 'calibration.json').read_text())
    history = json.loads((root / 'run95/training_history.json').read_text())
    checkpoint = torch.load(root / 'run95/agent95.pt', map_location='cpu',
                            weights_only=True)
    cfg = results['config']
    require(cfg == checkpoint['config'], 'Checkpoint and results configurations differ')
    require(results['parameters'] == checkpoint['parameters'] ==
            calibration['selected_parameters'], 'Parameter records differ')
    expected_configuration = {
        'maturity': 63, 'spot': 100.0, 'strike': 100.0, 'r': 0.0167, 'q': 0.0165,
        'confidence': 0.95, 'borrowing_limit': 100.0,
        'training_paths': 400000, 'validation_paths': 50000, 'test_paths': 100000,
        'epochs': 50, 'batch_size': 1000, 'learning_rate': 0.0005,
        'training_seed': 43, 'validation_seed': 44, 'test_seed': 45,
        'model_seed': 4301, 'price_seed': 46, 'initial_capital': 3.16,
    }
    require(cfg == expected_configuration, 'Configuration differs from the reported snapshot')
    require(len({cfg['training_seed'], cfg['validation_seed'], cfg['test_seed']}) == 3,
            'Physical simulation seeds are not distinct')
    require([row['epoch'] for row in history] == list(range(1, cfg['epochs'] + 1)),
            'Training history is incomplete or out of order')
    best = min(history, key=lambda row: row['validation_cvar'])
    require(checkpoint['epoch'] == results['best_epoch'] == best['epoch'],
            'Checkpoint does not match validation-based epoch selection')
    require(checkpoint['validation_cvar'] == results['best_validation_cvar'] ==
            best['validation_cvar'], 'Validation score records differ')
    require(all(torch.isfinite(tensor).all().item()
                for tensor in checkpoint['state_dict'].values()),
            'Checkpoint contains a non-finite policy parameter')

    with np.load(root / 'run95/terminal_results.npz', allow_pickle=False) as archive:
        require(set(archive.files) == {'deep_loss', 'delta_loss', 'overlay_pnl'},
                'Terminal archive has unexpected fields')
        deep, delta, overlay = (archive[name] for name in
                                ('deep_loss', 'delta_loss', 'overlay_pnl'))
        require(all(x.shape == (cfg['test_paths'],) and np.isfinite(x).all()
                    for x in (deep, delta, overlay)),
                'Terminal observations have invalid shape or values')
        require(np.allclose(deep, delta-overlay, rtol=0, atol=1e-8),
                'Stored terminal observations violate the overlay identity')
        point_estimates = [empirical_cvar(deep, cfg['confidence']),
                           empirical_cvar(deep, cfg['confidence']) -
                           empirical_cvar(delta, cfg['confidence']),
                           empirical_cvar(-overlay, cfg['confidence']),
                           float(np.mean(overlay))]
        published = [3.481, -0.081, 1.810, -0.254]
        require(len(results['metrics']) == 4, 'Expected four Table 1 quantities')
        for i, (row, estimate, target) in enumerate(zip(results['metrics'],
                                                       point_estimates, published)):
            require(row['published'] == target, f'Published target differs at metric {i+1}')
            require(abs(row['estimate']-estimate) <= 1e-12,
                    f'Stored point estimate differs at metric {i+1}')
        histogram = np.loadtxt(root / 'run95/figure1_histogram.csv',
                               delimiter=',', skiprows=1)
        bins = np.linspace(-5, 5, 101)
        counts, _ = np.histogram(overlay, bins=bins)
        require(histogram.shape == (100, 3) and
                np.allclose(histogram[:, 0], bins[:-1], rtol=0, atol=1e-12) and
                np.allclose(histogram[:, 1], bins[1:], rtol=0, atol=1e-12) and
                np.array_equal(histogram[:, 2], counts),
                'Histogram does not match the stored terminal observations')
        require(int(cfg['test_paths']-counts.sum()) == results['outside_central_plot'],
                'Central-range exclusion count differs')

    return {'passed': True, 'test_paths': cfg['test_paths'], 'best_epoch': checkpoint['epoch'],
            'recomputed_point_estimates': point_estimates}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(check_snapshot(args.root), indent=2))
