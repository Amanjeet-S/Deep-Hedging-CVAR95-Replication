# Deep hedging at 95% confidence and Figure 1, Panel B

Research and implementation by [Amanjeet Singh](https://github.com/Amanjeet-S).

Independent replication of the **95% CVaR agent** in Table 1 and the corresponding distribution in Figure 1 of *Is the difference between deep hedging and delta hedging a statistical arbitrage?*.

[Research note](docs/replication_report.pdf) · [LaTeX source](docs/replication_report.tex) · [Executed notebook](reproduction.ipynb)

## Results

The experiment uses 400,000 training paths, 50,000 separate validation paths and 100,000 independent test paths. Both hedges start with capital 3.16. The call has spot and strike 100 and maturity 63 trading days.

| Table 1 quantity | Published | Independent run | Empirical 95% interval |
|---|---:|---:|---:|
| Deep-hedging loss CVaR | 3.481 | 2.884726 | [2.790608, 2.978843] |
| Deep CVaR minus delta CVaR | −0.081 | −0.075220 | [−0.150799, 0.000358] |
| Difference-strategy loss CVaR | 1.810 | 1.865054 | [1.812472, 1.917635] |
| Difference-strategy mean P&L | −0.254 | −0.262780 | [−0.269389, −0.256170] |

The difference strategy holds deep-minus-delta positions with zero initial capital. Its positive loss CVaR does not satisfy the paper's statistical-arbitrage criterion in this sample. The interval for the estimated CVaR improvement includes zero.

The [histogram](run95/figure1_95.png) shows independently simulated outcomes. It is not the published Figure 1. The full view retains all observations; 244 lie outside the central range [−5, 5]. The supplied weights and [terminal outcomes](run95/terminal_results.npz) preserve the completed run.

These intervals condition on the selected policy, calibration and delta approximation. The necessary population tail moments are not established; the intervals are empirical diagnostics. The note provides the derivations and qualifications. The published calibration, seeds and original weights have not been recovered, and the numerical results differ.

## Figure 1, Panel B

I extend the comparison to the 85%, 90% and 95% agents using the same calibration, initial capital **3.16**, and 100,000 test paths. The two additional runs complete 50 epochs; validation selects epoch 47 for 85% and epoch 40 for 90%. The recorded 95% checkpoint remains epoch 31.

[Panel B](panel_b/figure1_panel_b.png) overlays the three independently simulated P&L distributions. [Full range](panel_b/figure1_panel_b_full_range.png) retains every outcome. Common central bins have width 0.1; the counts outside [−5, 5] are 712, 384 and 244, respectively.

| Agent | Deep loss CVaR | Deep minus delta CVaR | Overlay loss CVaR | Mean overlay profit |
|---|---:|---:|---:|---:|
| 85% | 1.337382 | -0.087177 | 1.119636 | 0.014390 |
| 90% | 1.875336 | -0.080143 | 1.416507 | -0.162795 |
| 95% | 2.884726 | -0.075220 | 1.865054 | -0.262780 |

CVaR is evaluated at each agent's confidence level. All three overlay loss CVaRs are positive in this sample; the 85% mean profit is slightly positive. The recomputed 95% deep losses agree exactly with the preserved baseline; the largest overlay difference is 4.3×10⁻¹⁴.

[Initial-capital audit](panel_b/capital_audit.json) confirms **3.16** in all saved configurations. With position paths fixed, changing both initial capitals translates both standalone loss CVaRs equally and leaves their gap and overlay profits unchanged. Reducing capital to the independently priced **2.770068** adds **0.391563** to each frozen-position CVaR; these frozen positions breach the cash limit on 33,680 paths for the 95% hedge, so that accounting control is not an admissible alternative policy.

Re-evaluating the saved 95% network from capital 2.770068 changes its wealth input and positions. Its loss CVaR becomes **3.419705**, its gap becomes **0.068196**, and overlay loss CVaR becomes **2.168302**. This is a fixed-weights policy re-evaluation, not retraining or a recovered published calibration. It does not reproduce the published difference statistics.

`panel_b.py` implements the extension. The [results](panel_b/panel_b_results.json), [common histogram](panel_b/panel_b_histogram.csv), selected checkpoints, complete training histories and terminal outcomes are in `panel_b/`. To retrain and evaluate:

```sh
python panel_b.py train --confidence 85 --folder panel_b_retrained
python panel_b.py train --confidence 90 --folder panel_b_retrained
python panel_b.py evaluate --folder panel_b_retrained --capital-control --full-range
```

## Calibration and initial variance

I tested the observed-residual forecast, a predefined initial state that prices the option near 3.16, and the percentage-unit likelihood fit. Initial capital remains **3.16** in every training and evaluation. The two new 95% policies complete 50 epochs and use validation-selected checkpoints; all test comparisons share the same 100,000 innovations.

| Setup | Call price, seed 46 | Deep CVaR95 | Delta CVaR95 | Gap |
|---|---:|---:|---:|---:|
| Published reference | 3.160 | 3.481 | 3.562 | −0.081 |
| Retained baseline | 2.770068 | 2.884726 | 2.959946 | −0.075220 |
| Observed forecast, baseline weights | 2.775015 | 2.900073 | 2.973967 | −0.073895 |
| Price-compatible state, retrained | 3.160828 | 4.010514 | 4.125525 | −0.115011 |
| Scaled fit, retrained | 3.052934 | 4.353688 | 4.429238 | −0.075551 |

The forecasting convention changes price by about 0.005. Aligning the option price does not recover the published hedging risk. The scaled fit moves both standalone risks together while leaving their gap close to the baseline. These are predefined sensitivity experiments; they do not identify the original published calibration.

[Risk comparison](calibration_sensitivity/calibration_risk_comparison.png) · [Physical variance moments](calibration_sensitivity/calibration_variance_moments.png) · [95% profit distributions](calibration_sensitivity/calibration_overlay_comparison.png)

Holding the baseline weights fixed in the price-compatible market raises deep CVaR by **1.351115**; retraining reduces it by **0.225326**, leaving a total change **1.125789**. The corresponding scaled-fit effects are **1.570174**, **−0.101212**, and **1.468962**. Fixed weights do not imply fixed positions when market and wealth inputs change.

Five admissible optimisation starts in percentage-return units agree within 4.3×10⁻¹¹ in log likelihood. Independent 500,000-path conditional-delta checks agree with the initial grid values within **0.334** and **0.798** sampling standard errors. All five evaluated policy rows pass accounting, finite-outcome and borrowing checks. The intervals remain empirical plug-in diagnostics with unestablished population coverage.

The public implementation's archived GJR file records price **3.157065** and 95% deep CVaR **3.024906**, which also differ from the published table. It does not retain the full-precision calibration, initial state or model weights. Directly stored archived values are distinguished from the paper and the independent simulations.

The complete inputs, selected weights, histories, terminal arrays, price diagnostics and numerical checks are in `calibration_sensitivity/`. To reproduce the two retrained controls in a new directory:

```sh
python calibration_sensitivity.py --folder sensitivity_retrained --prepare-only
python replication.py --calibration sensitivity_retrained/price_compatible/calibration.json --folder sensitivity_retrained/price_compatible/run95 --epochs 50 --device cpu
python replication.py --calibration sensitivity_retrained/scaled_fit/calibration.json --folder sensitivity_retrained/scaled_fit/run95 --epochs 50 --device cpu
python calibration_sensitivity.py --folder sensitivity_retrained
python calibration_plots.py --folder calibration_sensitivity
python independent_diagnostics.py --root . --folder diagnostic_rerun
python delta_sensitivity_mc.py --folder calibration_sensitivity
```

The delta grids are rebuilt in memory; an optional matching fine cache can accelerate the raw-coefficient cases. The calibration controls do not replace baseline or Panel B files.

## Method

The Gaussian GJR-GARCH model uses the known conditional variance of the **next return** at each decision. The selected raw-return fit uses 1,259 daily returns from 4 January 2016 to 31 December 2020. Its exact coefficients and the rescaled conditioning check are in [calibration.json](calibration.json).

The policy has four hidden layers of 56 ReLU units, Glorot initialisation and Adam optimisation. Training completes 50 epochs with batch size 1,000 and learning rate 0.0005. A borrowing limit of 100 constrains the deep hedge. The separately validated checkpoint is epoch 31; the delta benchmark is unconstrained.

Delta is computed by stock-numeraire Gaussian quadrature and interpolation, with an analytic final-day value and a wider grid for large variance states. [Validation](delta_validation.json) compares it with independent conditional Monte Carlo. This differs from the paper's nested Monte Carlo implementation.

The bounded put estimator and put–call parity give an initial call price of **2.770068**, using two million paths. A separate [price audit](stable_price_matching_audit.json) with the same coefficients and initial variance 0.0000775 gives **3.160828** and **3.161292** under two seeds. This illustrates compatibility with 3.16; the baseline and Panel B retain the fitted initial state. The separate initial-variance sensitivity below uses this price-compatible state.

## Reproduce

The recorded environment is Python 3.12.14 on macOS with CPU computation. Install the pinned dependencies from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_check.py
python verify_replication.py
```

The checks recompute the four statistics and verify agreement between outcomes, checkpoint, history, configuration and histogram, together with the financial identities and gradient. The notebook's default settings also refit the calibration and reprice the selected state. Install a Jupyter interface separately if required.

For a complete rerun:

```sh
python calibrate.py --data data/SP500_random.csv --output recalibration.json
python price_audit.py --output rerun-price-audit.json
python replication.py --calibration calibration.json --folder retrained95 --epochs 50 --device cpu
python evaluate.py --folder retrained95
```

These commands retain the recorded calibration for training. Inspect the refit before replacing it: the raw-return optimiser is sensitive to scaling. Training took approximately 23 minutes; evaluation and grid construction require further time and several GB of memory. Fixed seeds do not ensure identical training across hardware and libraries.

The seeds are 43 (training), 44 (validation), 45 (test), 4301 (initialisation), 1043 (shuffle) and 46 (pricing); seed 47 provides an independent price check. The complete configuration is in [run95/results.json](run95/results.json). `replication.py`, `delta_reference.py` and `evaluate.py` implement the market, policy, delta and evaluation. `run95/` contains all 50 epochs, the selected checkpoint, observations, statistics and figures. Evaluation rebuilds the omitted grid cache in memory.

Compile the standalone research note with a LaTeX installation that includes PGFPlots:

```sh
latexmk -pdf -outdir=docs docs/replication_report.tex
```

## Data and implementation provenance

The historical data and inspected implementation are from [cpmendoza/DeepHedging_StatisticalArbitrage](https://github.com/cpmendoza/DeepHedging_StatisticalArbitrage), commit [`ac76f29ab6fb9379df11b3e9087dd8023c2cbf5d`](https://github.com/cpmendoza/DeepHedging_StatisticalArbitrage/tree/ac76f29ab6fb9379df11b3e9087dd8023c2cbf5d). The historical file is copied from `data/raw/SP500_random.csv`; its MIT notice is retained in [AUTHORS_LICENSE.txt](AUTHORS_LICENSE.txt). The research implementation here is independently written.
