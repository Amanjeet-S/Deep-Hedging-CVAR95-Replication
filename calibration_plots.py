"""Plot the completed 95% calibration sensitivities and variance diagnostics."""
from pathlib import Path
import argparse, json, os

__author__ = 'Amanjeet Singh'

import numpy as np
import pandas as pd


def plots(root, folder):
    root, folder = Path(root), Path(folder)
    os.environ.setdefault('MPLCONFIGDIR', str(folder / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    report = json.loads((folder / 'sensitivity_results.json').read_text())
    baseline = json.loads((root / 'run95/results.json').read_text())
    cases = {c['name']: c for c in report['cases']}
    rows = [('Baseline', baseline)]
    for key, label in [('observed_forecast', 'Forecast\nfixed weights'),
                       ('price_compatible', 'Price-aligned\nretrained'),
                       ('scaled_fit', 'Scaled fit\nretrained')]:
        case = cases[key]
        row = next((p for p in case['policies'] if p['policy'] == 'retrained'), case['policies'][0])
        rows.append((label, row))

    def save(fig, stem, title):
        fig.savefig(folder / (stem + '.png'), dpi=220,
                    metadata={'Author': __author__, 'Title': title})
        fig.savefig(folder / (stem + '.svg'),
                    metadata={'Creator': __author__, 'Title': title})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = np.arange(len(rows))
    deep = [r['metrics'][0]['estimate'] for _, r in rows]
    delta = [r['delta_cvar'] for _, r in rows]
    a = ax.bar(x-.18, deep, .36, label='Deep hedge', color='#256c88')
    b = ax.bar(x+.18, delta, .36, label='Delta hedge', color='#7b8da2')
    ax.axhline(3.481, color='#a64135', linestyle='--', linewidth=1.4,
               label='Published deep CVaR: 3.481')
    ax.axhline(3.562, color='#935b88', linestyle=':', linewidth=1.4,
               label='Published implied delta CVaR: 3.562')
    ax.bar_label(a, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(b, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(x, [label for label, _ in rows])
    ax.set_ylabel('95% loss CVaR')
    ax.set_ylim(0, 5.3)
    ax.set_title('Hedging risk under calibration controls', fontsize=13)
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc='upper left')
    fig.text(.5, .025, 'Initial capital 3.16 throughout; common 100,000-path test innovations',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    save(fig, 'calibration_risk_comparison', '95 percent hedging risk under calibration controls')

    colours = {'baseline_fitted_state':'#279e68',
               'observed_last_residual_forecast':'#7b8da2',
               'illustrative_price_compatible_initial_variance':'#e69f00',
               'percentage_return_conditioning_fit_terminal_state':'#8259a6'}
    names = {'baseline_fitted_state':'Baseline',
             'observed_last_residual_forecast':'Observed forecast',
             'illustrative_price_compatible_initial_variance':'Price-aligned state',
             'percentage_return_conditioning_fit_terminal_state':'Scaled fit'}
    frame = pd.read_csv(folder / 'physical_variance_moments.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    for key, colour in colours.items():
        data = frame.loc[frame.scenario == key].sort_values('decision_date')
        style = '--' if key == 'observed_last_residual_forecast' else '-'
        axes[0].plot(data.decision_date, 100*np.sqrt(252*data.mean_variance),
                     color=colour, linestyle=style, label=names[key], linewidth=1.6)
        axes[1].plot(data.decision_date, 3*data.second_variance_moment,
                     color=colour, linestyle=style, label=names[key], linewidth=1.6)
    axes[0].set_ylabel('Annualised RMS volatility (%)')
    axes[1].set_ylabel('Fourth centred log-return moment')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Decision date'); ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle('Physical variance dynamics under calibration controls', fontsize=13)
    fig.text(.5, .02, 'Analytical moments; RMS = sqrt(252 E[h]); fourth return moment = 3 E[h²]',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .93))
    save(fig, 'calibration_variance_moments', 'Physical variance moments under calibration controls')

    fig, ax = plt.subplots(figsize=(9, 4.9))
    for name, label, colour in [('baseline','Baseline','#279e68'),
        ('price_compatible','Price-aligned state, retrained','#e69f00'),
        ('scaled_fit','Scaled fit, retrained','#8259a6')]:
        path = root/'run95/terminal_results.npz' if name=='baseline' else folder/name/'run95/terminal_results.npz'
        with np.load(path) as arrays: pnl = arrays['overlay_pnl']
        counts, edges = np.histogram(pnl, bins=np.linspace(-5,5,101))
        ax.stairs(counts, edges, fill=True, alpha=.12, color=colour)
        ax.stairs(counts, edges, color=colour, linewidth=1.5, label=label)
        ax.axvline(pnl.mean(), color=colour, linestyle=':', linewidth=1)
    ax.axvline(0, color='#17384f', linewidth=.8)
    ax.set_xlim(-5,5); ax.set_xlabel('Terminal P&L of the deep-minus-delta strategy')
    ax.set_ylabel('Frequency (paths)'); ax.grid(axis='y', alpha=.15)
    ax.set_title('95% agents under calibration controls', fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    fig.text(.5,.02,'Common bins of width 0.1; dotted lines mark means; all outcomes enter the statistics',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.075,1,1))
    save(fig, 'calibration_overlay_comparison', '95 percent difference-strategy calibration sensitivities')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--folder', type=Path, default=Path('calibration_sensitivity'))
    args = parser.parse_args()
    plots(args.root, args.folder)
