# Synthetic data suite: conventions for check authors

This file goes with `make_data.py` and the `*.npz` files and `MANIFEST.json` in this
directory. Every file is `<name>_T<T>.npz`, with `T` in {64, 256, 1024, 4096}. `x` has
shape (P, T), "ps" order (processes, observations), raw and not normalised. Load with
`make_data.load(name, T)`. It refuses to return data if `MANIFEST.json` is missing and
checks the sha256, dtype and shape of every array, plus the key set. There is no way to
turn verification off. The truth arrays describe how each dataset was built. They are not
values that any SPI has been shown to reproduce.

## Spectral matrices (D5 `var2_bivariate`, `var2_chain`)

* `S_true` (complex128, shape (513, P, P)) is the **two-sided** spectral density on
  `freqs = linspace(0, 0.5, 513)` (cycles/sample, spacing 1/1024):
  `S_true[f, i, j] = sum_h Gamma(h) e^{-2 pi i f h}` with `Gamma(h) = E[x_{t+h} x_t^T]`.
  Equivalently `S = H Sigma H^H`, with `H(f) = (I - A1 e^{-2 pi i f} - A2 e^{-4 pi i f})^{-1}`.
  Integrating over [-1/2, 1/2] gives `Gamma0`.
* `scipy.signal.csd(x_i, x_j)` computes `conj(X_i) X_j`. That matches
  **`S_true[:, j, i]`**, which equals `conj(S_true[:, i, j])`, not `S_true[:, i, j]`.
  Its default one-sided `scaling='density'` is **2 x `S_true`** at interior bins (DC and
  Nyquist are not doubled).
  * Magnitudes and coherence are the same under either index order. The sign of the
    phase and of the imaginary part flips.
  * Any other frequency grid must be matched to `freqs` or interpolated onto it.
    `S_true` is defined only on this grid.

## VAR coefficient convention (D5, D15)

`x_t = A1 x_{t-1} + A2 x_{t-2} + e_t` (D15: `x_t = A x_{t-1} + e_t`).
**`A[i, j]` is the effect of source j on target i (row = target, column = source).**
For example, `var2_bivariate` has `A1[1, 0] = 0.5`, meaning x0 -> x1. `var2_chain` is
x0 -> x1 -> x2, with no direct x0 -> x2 term.

## D15 `gauss_ar1_pair`: lagged covariance orientation

`Sigma_XY = E[X(t+1) X(t)^T] = A Sigma_X`. Rows index time t+1 and columns index time t.
A reference that defines Cov(past, future) as `Sigma_X A^T` gets the **transpose**. The
off-diagonals differ (0.9714 vs 0.8479), so transpose it before comparing. Other truth
keys:
* `Sigma_X`: the stationary covariance.
* `Sigma_E`: the innovation covariance, which equals Cov(X(t+1) | X(t)).

## D6 `delayed_noise`: lag sign

"x0 leads" means `x_k(t) = x0(t - d_k) + 0.1 noise`, with `delay = d = (0, 1, 3, 5)`.
An estimator that reports the lag of x_k relative to x0 should return **+d_k**. The delays
are not circular: the first d_k samples of x_k are x0 values from before the start of the
series, not wrapped-around samples.

## D7 `narrowband_f0-{0.1,0.25,0.4}`: phase sign

`x_k(t) = sin(2 pi f0 t + phase_shift_k) + 0.05 noise`, with
`phase_shift = (0, +pi/4, -pi/2)`.
* A **positive** shift means x_k **leads** x0 by `phase_shift_k / (2 pi f0)` samples.
* At f0 = 0.25, the -pi/2 process is exactly a 1-sample delay of x0.
* These are pure sinusoids plus white noise, not narrowband noise. Expect coherence of
  about 1 at f0. The Hilbert (analytic-signal) phase is `phase - pi/2`.

## D8 `logistic_maps`, `logistic_maps_bidir`, `logistic_maps_chaotic`, `logistic_maps_sugihara`: direction

Process 0 is x and process 1 is y. The update rules are
`x(t+1) = x(t) [r_x - r_x x(t) - beta_xy y(t)]` and
`y(t+1) = y(t) [r_y - r_y y(t) - beta_yx x(t)]`.
This is Sugihara et al. 2012, Eq. 1 of the main text. Its subscript order is target,
source.
* `beta_xy` multiplies y in x's update, so **`beta_xy > 0` means y -> x**.
* `beta_yx` multiplies x in y's update, so **`beta_yx > 0` means x -> y**.

| dataset | r_x | r_y | beta_xy (y->x) | beta_yx (x->y) | role |
|---|---|---|---|---|---|
| `logistic_maps` | 3.8 | 3.5 | 0.1 | 0 | y -> x; the undriven y is a **period-4 orbit** (4 distinct values). Kept as the **tied-data fixture**: it exercises nearest-neighbour ties (CCM `std::sort` order, KSG/Kozachenko jitter). Not a direction-check dataset. |
| `logistic_maps_bidir` | 3.8 | 3.5 | 0.1 | 0.02 | bidirectional, y -> x stronger; both series aperiodic |
| `logistic_maps_chaotic` | 3.8 | 3.5 | 0 | 0.1 | **x -> y only**; the chaotic r = 3.8 map drives the r = 3.5 map, so both series are aperiodic. Use this for CCM / kNN-TE direction checks. |
| `logistic_maps_sugihara` | 3.8 | 3.5 | 0.02 | 0.1 | Sugihara et al. 2012 Fig. 1 caption parameters (x -> y stronger) |

Direction expectations on the pyspi matrix follow from the convention table of each
family (CCM: skill of "a xmap b" is high when b causes a; TE: TE(b -> a)). For
`logistic_maps_chaotic` the x -> y coupling is the only one present, so any directed
measure must favour x -> y over y -> x; for `logistic_maps_sugihara` x -> y must be the
stronger of the two.

## Which truth goes with which pyspi path

* **D3 `scaled_offset`:** `cov_true` (= `outer(scale, scale) * D2 cov`) applies to the raw
  `x`. pyspi's pipeline path z-scores first, so on that path compare against `corr_true`
  (correlation) instead.
* **D4 `trending`:** `cov_true_detrended` (= D2's unit-variance covariance) applies after
  linear detrending (pipeline-detrend), not to the raw trending `x`.
* **D10 `contaminated`:**
  * `clean_cov_true` is the target for robust estimators.
  * `mixture_cov_population` is the target for non-robust estimators. It is computed with
    the **realised** outlier fraction `outlier_mask.sum() / T`, for example 10/64 at T = 64
    rather than 0.15.
  * `x_clean` is the series before contamination.
* **D13 `sparse_precision`:** `cov_true` does not have unit variance (its diagonal runs
  from 1.25 to 1.62). Compare:
  * precision SPIs against `precision_true` (tridiagonal: 1.0 on the diagonal, -0.4 off
    it);
  * covariance SPIs against `cov_true`;
  * correlation SPIs against the normalised `cov_true` (`C / sqrt(outer(diag C, diag C))`).

## D11 `cointegrated`

`coint_vector = [2, -1, 0, 0]`: `2 x0 - x1` is stationary. The Engle–Granger slope of x1
regressed on x0 is 2. x2 and x3 are independent random walks.
`is_cointegrated[i, j]` is symmetric and False on the diagonal. `common_trend` is w.

## D14 `shared_envelope`

* `phase` is the **wrapped argument of sin**, not the analytic-signal phase, which is
  about `phase - pi/2`.
* Carrier: 0.2 cycles/sample. Envelope: 1 + 0.5 sin(2 pi 0.005 t).
* x1 lags x0 by pi/3 in phase.
* **`phase_offset_rel_x0[3]` is NaN** because x3 has its own random-walk phase and envelope,
  so it has no constant offset to x0. `phase_offset_defined = [True, True, True, False]`
  marks which entries are meaningful. Mask with it rather than testing `isnan`.
* **x2 is bit-identical to x0**, so the (0, 2) pair is degenerate. Its correlation is
  exactly 1 and its differences have zero variance. Expect NaN, inf or warnings from some
  SPIs. The pair is useful as a null case for symmetry and idempotence checks.

## Seeds and draws

* Every file uses `np.random.default_rng(base_seed * 1_000_000 + T)`.
* Different T are **independent draws, not prefixes** of each other.
* D3, D4 and D10 reuse D2's draw (`bivariate_gaussian`), so their manifest `seed` is D2's.
* `own_seed` is listed for all three but is used only for D10's contamination (outlier
  positions and values).

## Environment and byte identity

* Byte identity across the two venvs depends on **numpy 2.5.4 in both** (the project
  `.venv` and the audit venv). numpy does not guarantee that the Generator stream stays the
  same across versions (NEP 19).
* `MANIFEST.json` records `numpy_version`, and the per-array hashes catch any change. If
  `load()` raises on a different numpy version, regenerate in both venvs and treat any
  results computed before as stale.

## Datasets with no numeric truth beyond their construction

* **D1 `iid_gaussian`:** 5 iid N(0, 1) processes. The expected value is the null
  (independence, symmetry).
* **D9 `nonlinear_dependence`:**
  * `x1 = x0^2 + 0.05 N(0, 1)` with `x0 ~ U(-1, 1)`. Pearson correlation is about 0, but
    x0 and x1 are strongly dependent.
  * `(x2, x3) = (cos theta, sin theta)` with `theta ~ U(0, 2 pi)` (stored as `theta`), so
    `x2^2 + x3^2 = 1`.
* **D12 `anm`:** additive-noise models `x1 = x0^3 + 0.2 N(0, 1)` and
  `x3 = tanh(2 x2) + 0.2 N(0, 1)`, with x0 and x2 ~ U(-1, 1). The index arrays give the
  direction: `cause = [0, 2]` and `effect = [1, 3]` (cause[k] -> effect[k]).
