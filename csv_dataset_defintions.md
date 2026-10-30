# PNW corrected training CSV

`pnw_training_corrected.csv`: **38,579 event rows**.

## Rows and labels

One row represents one event. Each feature is calculated for individual station recordings, then summarized by the median of the available values.

- `event_id`: event identifier for joining files.
- `label`: **0** = earthquake; **1** = catalog-labeled explosion.
- `geographic_fold`: one of five evaluation sets. Events in the same geographic group stay together.
- Columns ending in `__n_stations` count the stations contributing to that feature.

Each recording contains about **150 seconds** at **100 samples per second** in three directions: **E** (east–west), **N** (north–south), and **Z** (vertical).

**P** and **S** are the supplied arrival times. A **window** is the part of a recording used for a measurement.

RMS measures typical amplitude. For n samples:

```math
\mathrm{RMS}(x)=\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}
```

## Waveform preparation

For the first five features, subtract each component's mean over the entire recording. Smoothly reduce the amplitude toward zero over the first and last 5%, without removing samples.

Each feature uses its own frequency filter. The filters are not applied one after another. Bandpass filters are fourth-order Butterworth filters applied forward and backward.

## P/S amplitude ratio

Column: `nb_log10_pz_sh_rms_ratio_4_8hz`

Compares vertical motion near P with horizontal motion near S.

```math
R_{PS}=\log_{10}\left(\frac{\mathrm{RMS}(Z_P)}{\mathrm{RMS}(H_S)}\right)
```

Horizontal amplitude combines the east–west and north–south signals:

```math
H=\sqrt{E^2+N^2}
```

Subscripts P and S identify the measurement windows.

- Filter: **4–8 Hz**.
- P window: starts at P and ends after **1.5 seconds**, or at S if S arrives sooner. It must last at least **0.3 seconds**.
- S window: **S to S+3 seconds**.

## S horizontal/vertical ratio

Column: `nb_log10_sh_sz_rms_ratio_4_8hz`

Compares horizontal and vertical motion during the S window.

```math
R_{HV}=\log_{10}\left(\frac{\mathrm{RMS}(H_S)}{\mathrm{RMS}(Z_S)}\right)
```

- Filter: **4–8 Hz**.
- Window: **S to S+3 seconds**.

## Amplitude rise time

Column: `nb_envelope_rise_10_90_s_2_10hz`

Measures the time between the first crossings of 10% and 90% of the largest smoothed amplitude, searching only up to that peak.

```math
T_{\mathrm{rise}}=t_{90}-t_{10}
```

Combine the three filtered components, then smooth the resulting amplitude with a **0.2-second moving average**. This smoothed signal is the envelope.

```math
A=\sqrt{E^2+N^2+Z^2}
```

- Filter: **2–10 Hz**.
- Window: **P−1 to S+10 seconds**.
- Crossing times use the recorded sample times, without interpolation.
- Units: **seconds**.

## Envelope excess kurtosis

Column: `nb_envelope_excess_kurtosis_2_10hz`

Measures how extreme the envelope's values are relative to their usual variation. Here e is the envelope, and its mean is denoted by an overbar.

```math
K_{\mathrm{excess}}=
\frac{\mathrm{mean}\left[(e-\bar{e})^4\right]}
{\left(\mathrm{mean}\left[(e-\bar{e})^2\right]\right)^2}-3
```

Uses the same filter, smoothing, and window as rise time. Calculated with `scipy.stats.kurtosis(e, fisher=True, bias=True)`.

## Low/high frequency power ratio

Column: `nb_log10_psd_power_1_5_over_5_15hz`

Compares power in a lower-frequency band with power in a higher-frequency band.

```math
R_{\mathrm{power}}=\log_{10}\left(\frac{P_{\mathrm{low}}}{P_{\mathrm{high}}}\right)
```

- Low band: **1 Hz up to, but excluding, 5 Hz**.
- High band: **5 Hz up to, but excluding, 15 Hz**.
- Window: **P−1 to P+20 seconds**.
- No bandpass filter before spectral estimation.

Welch's method estimates each component's power spectrum using **2-second segments**. Sum the three component spectra, then integrate each band with the trapezoidal rule.

## Crest factor

Column: `cb_crest_factor`

Measures the largest absolute signal peak relative to its typical amplitude.

```math
C=\frac{\max_i |x_i|}{\mathrm{RMS}(x)}
```

Remove the fitted straight-line baseline from each entire component recording, then filter to **1–15 Hz**.

- Window: **P−5 to P+40 seconds**.
- Calculate separately for each component, then take their median.

## Spectral entropy

Column: `cb_spectral_entropy`

Measures how evenly power is spread across frequencies. Lower values mean power is concentrated at fewer frequencies; higher values mean a more even spread.

Let Q be the spectral power values, K the number of retained frequency bins, and p each bin's share of the total power.

```math
p_k=\frac{Q_k}{\sum_{j=1}^{K}Q_j}
```

```math
H=-\frac{\sum_{k=1}^{K}p_k\ln\left(\max(p_k,10^{-15})\right)}{\ln K}
```

Uses the same baseline removal and **1–15 Hz** filter as crest factor.

- Window: **P to P+40 seconds**.
- Welch segments: **4 seconds**.
- Retained frequencies: **1 Hz up to, but excluding, 15 Hz**.
- Calculate separately for each component, then take their median.
