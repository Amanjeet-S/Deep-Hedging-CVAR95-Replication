"""Reproduce the initial-call-price sensitivity audit without training agents.

Run: python price_audit.py
The two h=7.75e-5 cases are price-compatible illustrations selected after a
bracket; they do not recover the authors' original initial variance.
"""
from __future__ import annotations

__author__ = "Amanjeet Singh"
import argparse
import json
from pathlib import Path

from replication import Config, risk_neutral_price


def run(calibration_path: Path, output_path: Path, n: int):
    calibration = json.loads(calibration_path.read_text())
    base = dict(calibration['selected_parameters'])
    h0 = base['initial_variance']
    specifications = []
    for multiple in (1, 2, 3, 4):
        params = dict(base, initial_variance=h0*multiple)
        specifications.append((f'source_coeffs_h0_multiple_{multiple}', params, 46))
    rounding = dict(base, mu=.000625, alpha=.1125, gamma=.2025, beta=.7825)
    specifications.append(('rounding_witness_half_upper_interval', rounding, 46))
    for seed in (46, 47):
        illustrative = dict(base, initial_variance=7.75e-5)
        specifications.append(('illustrative_unreported_initial_variance_7p75e_minus5', illustrative, seed))

    scenarios = []
    for label, params, seed in specifications:
        result = risk_neutral_price(params, Config(price_seed=seed), n=n)
        price = result['price']
        se = result['standard_error']
        row = dict(result,
                   label=label,
                   parameters={k: v for k, v in params.items() if k != 'initial_variance'},
                   first_return_conditional_variance=params['initial_variance'],
                   standard_error_antithetic_pairs=se,
                   ci95=[price-1.96*se, price+1.96*se],
                   seed=seed,
                   price_3p16_in_ci95=bool(price-1.96*se <= 3.16 <= price+1.96*se))
        scenarios.append(row)
        print(json.dumps({'label': label, 'seed': seed, 'price': price,
                          'standard_error': se, 'zero_stock_absorptions': result['absorbed_numerically_zero_stock_paths']}), flush=True)

    out = {
        'purpose': 'Initial-state/rounding price sensitivity; illustrative scenarios are not recovered original calibration',
        'method': 'Paper Q dynamics:62 simulated periods plus exact final one-period conditional Gaussian put; put-call parity;1million antithetic pairs at default2million paths',
        'guard': 'A path is absorbed only when actual float64 exp(logspot)==0, and receives the full discounted-strike put contribution. No path is dropped and no positive variance is clipped.',
        'rates': {'r': .0167, 'q': .0165},
        'S0': 100, 'K': 100, 'T_days': 63,
        'source': calibration_path.name,
        'illustrative_state_warning': '7.75e-5 was selected after the initial-state bracket solely to demonstrate an attainable callprice; it is not an observed fitted variance or a recovered original parameter.',
        'scenarios': scenarios,
    }
    output_path.write_text(json.dumps(out, indent=2)+'\n')
    return out


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=here/'calibration.json')
    parser.add_argument('--output', type=Path, default=here/'stable_price_matching_audit.json')
    parser.add_argument('--paths', type=int, default=2000000)
    args = parser.parse_args()
    run(args.calibration, args.output, args.paths)
