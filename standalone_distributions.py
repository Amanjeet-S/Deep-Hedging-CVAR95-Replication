"""Plot standalone terminal net hedge P&Ls on their original price scale.

Run from any directory. Optional --root selects a reproduction directory and
--tex-snippet writes inline PGFplots coordinates for the central comparison.
"""
from __future__ import annotations

__author__ = "Amanjeet Singh"

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ORDER = ('deep85', 'deep90', 'deep95', 'delta')
LABELS = {'deep85': 'Deep hedge, CVaR 85%', 'deep90': 'Deep hedge, CVaR 90%',
          'deep95': 'Deep hedge, CVaR 95%', 'delta': 'Delta hedge'}
COLOURS = {'deep85': '#1f77b4', 'deep90': '#17becf', 'deep95': '#279e68', 'delta': '#d97925'}
QUANTILES = (0., .0001, .001, .005, .01, .025, .05, .1, .25, .5, .75, .9, .95, .975, .99, .995, .999, .9999, 1.)
BIN_EDGES = np.linspace(-5., 5., 101)


def read_samples(root):
    summary = json.loads((root / 'panel_b/panel_b_results.json').read_text())
    for tag in ('85', '90', '95'):
        assert summary['agents'][tag]['config']['initial_capital'] == 3.16
        assert summary['agents'][tag]['config']['spot'] == 100.
        assert summary['agents'][tag]['config']['maturity'] == 63
    assert summary['common_market_paths'] and summary['test_paths'] == 100000
    with np.load(root / 'panel_b/panel_b_terminal_results.npz', allow_pickle=False) as stored:
        panel = {key: stored[key].copy() for key in stored.files}
    with np.load(root / 'run95/terminal_results.npz', allow_pickle=False) as stored:
        original = {key: stored[key].copy() for key in stored.files}
    samples = {f'deep{tag}': -panel[f'deep_loss{tag}'] for tag in ('85', '90', '95')}
    samples['delta'] = -panel['delta_loss']
    checks = {'common_path_count': 100000, 'all_samples_finite': True,
              'delta_saved_baseline_max_error': float(np.max(np.abs(panel['delta_loss'] - original['delta_loss']))),
              'deep95_saved_baseline_max_error': float(np.max(np.abs(panel['deep_loss95'] - original['deep_loss']))),
              'overlay95_saved_baseline_max_error': float(np.max(np.abs(panel['overlay_pnl95'] - original['overlay_pnl']))),
              'standalone_wealth_minus_payoff_max_errors': {}, 'standalone_difference_overlay_max_errors': {}}
    checks['standalone_wealth_minus_payoff_max_errors']['delta'] = float(np.max(np.abs(samples['delta'] - (panel['delta_wealth'] - panel['payoff']))))
    for tag in ('85', '90', '95'):
        name = f'deep{tag}'
        checks['standalone_wealth_minus_payoff_max_errors'][name] = float(np.max(np.abs(samples[name] - (panel[f'deep_wealth{tag}'] - panel['payoff']))))
        checks['standalone_difference_overlay_max_errors'][name] = float(np.max(np.abs(samples[name] - samples['delta'] - panel[f'overlay_pnl{tag}'])))
    assert all(len(samples[name]) == 100000 and np.isfinite(samples[name]).all() for name in ORDER)
    assert checks['delta_saved_baseline_max_error'] < 1e-8
    assert checks['deep95_saved_baseline_max_error'] < 1e-8
    assert checks['overlay95_saved_baseline_max_error'] < 1e-8
    assert max(checks['standalone_wealth_minus_payoff_max_errors'].values()) < 1e-8
    assert max(checks['standalone_difference_overlay_max_errors'].values()) < 1e-8
    return samples, checks


def describe(values):
    n = len(values)
    counts, _ = np.histogram(values, bins=BIN_EDGES)
    below = int(np.count_nonzero(values < BIN_EDGES[0]))
    above = int(np.count_nonzero(values > BIN_EDGES[-1]))
    assert int(counts.sum()) + below + above == n
    density = counts / (n * np.diff(BIN_EDGES))
    return {
        'n': n, 'finite': bool(np.isfinite(values).all()),
        'mean': float(values.mean()), 'median': float(np.median(values)),
        'sample_standard_deviation': float(values.std(ddof=1)),
        'minimum': float(values.min()), 'maximum': float(values.max()),
        'quantiles': {f'{q:g}': float(np.quantile(values, q)) for q in QUANTILES},
        'negative_probability': float(np.mean(values < 0)),
        'zero_probability': float(np.mean(values == 0)),
        'positive_probability': float(np.mean(values > 0)),
        'below_minus5': below, 'above_plus5': above,
        'outside_central_range': below + above,
        'inside_central_range': int(counts.sum()),
        'central_probability_mass': float(counts.sum() / n),
    }, counts, density


def save_tables(folder, statistics, counts, densities):
    fields = ['strategy', 'n', 'finite', 'mean', 'median', 'sample_standard_deviation',
              'minimum', 'maximum', 'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99',
              'negative_probability', 'zero_probability', 'positive_probability',
              'below_minus5', 'above_plus5', 'outside_central_range',
              'inside_central_range', 'central_probability_mass']
    with (folder / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in ORDER:
            record = {key: statistics[name][key] for key in fields if key != 'strategy' and not key.startswith('q')}
            record['strategy'] = LABELS[name]
            for field, q in [('q01', '.01'), ('q05', '.05'), ('q25', '.25'), ('q50', '.5'), ('q75', '.75'), ('q95', '.95'), ('q99', '.99')]:
                record[field] = statistics[name]['quantiles'][f'{float(q):g}']
            writer.writerow(record)
    columns = [BIN_EDGES[:-1], BIN_EDGES[1:]]
    header = ['bin_left', 'bin_right']
    for name in ORDER:
        columns.extend([counts[name], densities[name]])
        header.extend([f'{name}_count', f'{name}_density'])
    np.savetxt(folder / 'central_histogram.csv', np.column_stack(columns), delimiter=',',
               header=','.join(header), comments='', fmt='%.16g')
    with (folder / 'quantiles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['quantile', *ORDER])
        for q in QUANTILES:
            writer.writerow([q, *[statistics[name]['quantiles'][f'{q:g}'] for name in ORDER]])


def save_figure(fig, folder, filename, title):
    fig.savefig(folder / f'{filename}.png', dpi=210,
                metadata={'Author': __author__, 'Title': title})
    svg_path = folder / f'{filename}.svg'
    fig.savefig(svg_path,
                metadata={'Creator': __author__, 'Title': title, 'Date': None})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')


def draw(folder, samples, statistics, densities):
    config_temp = tempfile.TemporaryDirectory(prefix='standalone-')
    os.environ.setdefault('MPLCONFIGDIR', config_temp.name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    common_peak = max(float(densities[name].max()) for name in ORDER) * 1.12
    x_label = r'Terminal net hedge P&L (price units; $S_0=100$)'

    def central_axis(ax, names, title, show_y=True):
        for name in names:
            ax.stairs(densities[name], BIN_EDGES, color=COLOURS[name], fill=True, alpha=.10)
            ax.stairs(densities[name], BIN_EDGES, color=COLOURS[name], linewidth=1.8, label=LABELS[name])
            ax.axvline(statistics[name]['mean'], color=COLOURS[name], linestyle='--', linewidth=1.15, alpha=.95)
        ax.axvline(0, color='#111111', linewidth=1.05)
        ax.set(xlim=(-5, 5), ylim=(0, common_peak), title=title, xlabel=x_label)
        if show_y:
            ax.set_ylabel('Full-sample empirical density')
        ax.grid(axis='y', alpha=.16)
        ax.legend(loc='upper left', frameon=False, fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(12., 5.1), sharex=True, sharey=True)
    central_axis(axes[0], ORDER[:3], 'Deep hedges')
    central_axis(axes[1], ('delta',), 'Common delta hedge', False)
    fig.suptitle('Standalone terminal hedge P&L distributions', fontsize=15, fontweight='bold')
    fig.text(.5, .024, '100,000 common physical-measure outcomes; initial capital 3.16; dashed lines mark means\n'
             'Net P&L = terminal hedge wealth − option payoff. Central bins retain the full-sample denominator.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    save_figure(fig, folder, 'standalone_central_comparison', 'Standalone terminal hedge P&L distributions')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12., 8.3), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, ORDER):
        central_axis(ax, (name,), LABELS[name])
        ax.axvline(statistics[name]['median'], color=COLOURS[name], linestyle=':', linewidth=1.2)
        stat = statistics[name]
        ax.text(.98, .96, f"Mean {stat['mean']:+.3f}\nMedian {stat['median']:+.3f}\nNegative: {100 * stat['negative_probability']:.2f}%\nOutside ±5: {stat['outside_central_range']:,}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .90})
    fig.suptitle('Separate standalone hedge P&L distributions', fontsize=15, fontweight='bold')
    fig.text(.5, .023, 'Black line: zero. Dashed line: mean. Dotted line: median.\n'
             'All panels use common axes and the full sample for density normalisation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .945))
    save_figure(fig, folder, 'standalone_central_panels', 'Separate standalone hedge P&L distributions')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.7, 5.5))
    for name in ORDER:
        sorted_values = np.sort(samples[name])
        cumulative = np.arange(1, len(sorted_values) + 1, dtype=np.float64) / len(sorted_values)
        ax.step(sorted_values, cumulative, where='post', color=COLOURS[name], linewidth=1.65, label=LABELS[name])
    low = min(statistics[name]['minimum'] for name in ORDER)
    high = max(statistics[name]['maximum'] for name in ORDER)
    ax.set_xscale('symlog', linthresh=5, linscale=1)
    ax.set_yscale('log')
    ax.set_xlim(low * 1.04, high * 1.06)
    ax.set_ylim(1 / len(samples['delta']), 1.08)
    ticks = [-60, -40, -20, -5, 0, 5, 10, 20]
    ax.set_xticks([value for value in ticks if low * 1.04 <= value <= high * 1.06])
    ax.set_xticklabels([str(value) for value in ticks if low * 1.04 <= value <= high * 1.06])
    ax.set_yticks([.00001, .0001, .001, .01, .1, 1.])
    ax.set_yticklabels(['0.001%', '0.01%', '0.1%', '1%', '10%', '100%'])
    ax.axvline(0, color='#111111', linewidth=1.05)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Empirical cumulative probability (log scale)')
    ax.set_title('Full-range empirical distributions, including the tails', fontsize=14, fontweight='bold')
    ax.grid(alpha=.16, which='major')
    ax.legend(frameon=False, loc='lower right')
    fig.text(.5, .021, 'Unshifted empirical CDF; every observed outcome contributes\n'
             'The horizontal scale is linear between −5 and 5 and logarithmic beyond that range.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .98))
    save_figure(fig, folder, 'standalone_full_range_ecdf', 'Full-range standalone hedge P&L empirical distributions')
    plt.close(fig)


def write_tex(path, densities, statistics):
    path.parent.mkdir(parents=True, exist_ok=True)
    ymax = max(float(densities[name].max()) for name in ORDER) * 1.12
    colours = {'deep85': 'standDeepA', 'deep90': 'standDeepB', 'deep95': 'standDeepC', 'delta': 'standDelta'}
    lines = ['% Author: Amanjeet Singh', '% Requires tikz and pgfplots.',
             '\\definecolor{standDeepA}{HTML}{1F77B4}', '\\definecolor{standDeepB}{HTML}{17BECF}',
             '\\definecolor{standDeepC}{HTML}{279E68}', '\\definecolor{standDelta}{HTML}{D97925}',
             '\\begin{tikzpicture}']
    shared = f'width=.475\\linewidth,height=5.4cm,xmin=-5,xmax=5,ymin=0,ymax={ymax:.8g},'
    shared += 'xlabel={Terminal net hedge P\\&L},grid=major,grid style={gray!15},'
    shared += 'tick label style={font=\\scriptsize},label style={font=\\small},title style={font=\\small},'
    shared += 'legend style={font=\\scriptsize,draw=none,fill=none,at={(0.02,0.98)},anchor=north west}'
    for axis_name, names, title in [('standdeep', ORDER[:3], 'Deep hedges'), ('standdelta', ('delta',), 'Common delta hedge')]:
        location = '' if axis_name == 'standdeep' else ',at={(standdeep.east)},anchor=west,xshift=.03\\linewidth,yticklabels={}'
        ylabel = ',ylabel={Full-sample empirical density}' if axis_name == 'standdeep' else ''
        lines.append(f'\\begin{{axis}}[name={axis_name},{shared},title={{{title}}}{ylabel}{location}]')
        for name in names:
            coordinates = ' '.join(f'({x:.8g},{y:.10g})' for x, y in zip(BIN_EDGES[:-1], densities[name])) + ' (5,0)'
            lines.append(f'\\addplot[const plot,no marks,color={colours[name]},line width=.9pt] coordinates {{{coordinates}}};')
            legend = LABELS[name].replace('%', '\\%').replace('Deep hedge, ', '')
            lines.append(f'\\addlegendentry{{{legend}}}')
            mean = statistics[name]['mean']
            lines.append(f'\\addplot[forget plot,no marks,dashed,color={colours[name]},line width=.55pt] coordinates {{({mean:.10g},0) ({mean:.10g},{ymax:.8g})}};')
        lines.append(f'\\addplot[forget plot,no marks,black,line width=.6pt] coordinates {{(0,0) (0,{ymax:.8g})}};')
        lines.append('\\end{axis}')
    lines.extend(['\\end{tikzpicture}', '% Density=count/(100000*bin width); unchanged net P&L; S0=100 and initial capital 3.16.'])
    path.write_text('\n'.join(lines) + '\n')


def run(root, folder=None, tex_snippet=None):
    root = Path(root).resolve()
    folder = Path(folder).resolve() if folder else root / 'standalone_distributions'
    folder.mkdir(parents=True, exist_ok=True)
    samples, checks = read_samples(root)
    statistics, counts, densities = {}, {}, {}
    for name in ORDER:
        statistics[name], counts[name], densities[name] = describe(samples[name])
    results = {
        'author': __author__, 'quantity': 'Terminal net hedge P&L = V_T - H = -hedging loss',
        'measure': 'Physical measure', 'initial_capital': 3.16, 'normalised_initial_spot': 100.,
        'maturity_days': 63, 'sample_count': 100000, 'common_market_outcomes': True,
        'centres': 'Original unshifted samples with zero shown as a reference point.',
        'statistics_note': 'Means, standard deviations, sign probabilities and quantiles are empirical summaries of the finite saved sample.',
        'histogram': {'range': [-5, 5], 'bins': 100, 'bin_width': .1,
                      'density_definition': 'count/(full sample count * bin width)',
                      'outside_outcomes_retained_in_denominator': True},
        'strategies': {name: {'label': LABELS[name], **statistics[name]} for name in ORDER}, 'checks': checks,
    }
    (folder / 'results.json').write_text(json.dumps(results, indent=2))
    save_tables(folder, statistics, counts, densities)
    draw(folder, samples, statistics, densities)
    if tex_snippet:
        write_tex(Path(tex_snippet), densities, statistics)
    print(json.dumps({'author': __author__, 'strategies': results['strategies'], 'checks': checks}, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--tex-snippet', type=Path)
    arguments = parser.parse_args()
    run(arguments.root, arguments.folder, arguments.tex_snippet)
