# Deep hedging at 95% confidence

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

## Method

The Gaussian GJR-GARCH model uses the known conditional variance of the **next return** at each decision. The selected raw-return fit uses 1,259 daily returns from 4 January 2016 to 31 December 2020. Its exact coefficients and the rescaled conditioning check are in [calibration.json](calibration.json).

The policy has four hidden layers of 56 ReLU units, Glorot initialisation and Adam optimisation. Training completes 50 epochs with batch size 1,000 and learning rate 0.0005. A borrowing limit of 100 constrains the deep hedge. The separately validated checkpoint is epoch 31; the delta benchmark is unconstrained.

Delta is computed by stock-numeraire Gaussian quadrature and interpolation, with an analytic final-day value and a wider grid for large variance states. [Validation](delta_validation.json) compares it with independent conditional Monte Carlo. This differs from the paper's nested Monte Carlo implementation.

The bounded put estimator and put–call parity give an initial call price of **2.770068**, using two million paths. A separate [price audit](stable_price_matching_audit.json) with the same coefficients and initial variance 0.0000775 gives **3.160828** and **3.161292** under two seeds. This illustrates compatibility with 3.16; that variance was not used for training and does not recover the original state.

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
