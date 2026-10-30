# PNW corrected training CSV

`pnw_training_corrected.csv`: **38,579 event rows**.

## Rows and labels

One row per event. Each feature is measured at individual stations, then summarized by the median of the available finite values.

- `event_id`: event identifier for joining files.
- `label`: **0** = earthquake (`eq`); **1** = explosion (`px/ex/qb`).
- `geographic_fold`: marks one of five evaluation sets. Events in the same geographic group stay in the same set.
- Each `__n_stations` column counts stations contributing a finite feature value.

Each station recording contains about **150 seconds** at **100 samples per second** in three directions: **E** (east–west), **N** (north–south), and **Z** (vertical). **P** and **S** are the supplied P-wave and S-wave arrival times. A window is the portion of a recording used for a measurement.

RMS measures typical amplitude. For $n$ signal samples $x_i$,

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}
$$

## Waveform features

For the first five features, subtract each component’s mean over the entire recording. Then taper the first and last 5%: smoothly reduce amplitude toward zero at the ends, without removing samples.

A bandpass filter retains a chosen frequency range. Each feature uses its specified filter separately; filters are not chained. Filters are fourth-order Butterworth, applied forward and backward.

### nb_log10_pz_sh_rms_ratio_4_8hz — P/S amplitude ratio

Compares vertical motion near P with horizontal motion near S. Horizontal amplitude is $H=\sqrt{E^2+N^2}$; subscripts identify the P or S window.

$$
\text{log\_ps}=\log_{10}\!\left(\frac{\operatorname{RMS}(Z_P)}{\operatorname{RMS}(H_S)}\right)
$$

Filter: **4–8 Hz**. P window: starts at P, duration **min(1.5 seconds, S−P)**, requiring at least 0.3 seconds. S window: **S to S+3 seconds**.

### nb_log10_sh_sz_rms_ratio_4_8hz — S horizontal/vertical ratio

Compares horizontal and vertical motion during S.

$$
\text{log\_hv\_s}=\log_{10}\!\left(\frac{\operatorname{RMS}(H_S)}{\operatorname{RMS}(Z_S)}\right)
$$

Filter: **4–8 Hz**. Window: **S to S+3 seconds**.

### nb_envelope_rise_10_90_s_2_10hz — amplitude rise time

Elapsed time between the first 10% and 90% crossings of the largest smoothed amplitude in the selected window. Search only up to the peak; use sample times without interpolation. The smoothed three-component amplitude is called the envelope.

$$
\text{rise\_time}=t_{90\%}-t_{10\%}
$$

Filter: **2–10 Hz**. Amplitude: $\sqrt{E^2+N^2+Z^2}$, smoothed with a **0.2-second moving average**. Window: **P−1 to S+10 seconds**.

### nb_envelope_excess_kurtosis_2_10hz — envelope excess kurtosis

Describes how extreme the smoothed amplitude values are relative to their usual variation. $e$ is the smoothed amplitude; $\bar e$ is its mean.

$$
\text{env\_kurtosis}=
\frac{\operatorname{mean}\!\left[(e-\bar e)^4\right]}
{\left(\operatorname{mean}\!\left[(e-\bar e)^2\right]\right)^2}-3
$$

Uses the rise-time filter, smoothing, and window; `scipy.stats.kurtosis(e)` defaults: `fisher=True, bias=True`.

### nb_log10_psd_power_1_5_over_5_15hz — low/high frequency power ratio

Compares power in lower and higher frequency bands.

$$
\text{log\_lowhigh}=\log_{10}\!\left(\frac{\text{power}_{1\leq f\lt 5\,\mathrm{Hz}}}{\text{power}_{5\leq f\lt 15\,\mathrm{Hz}}}\right)
$$

Window: **P−1 to P+20 seconds**. No bandpass. Welch’s method estimates power per frequency using **2-second segments**. Sum the three component spectra, then integrate each band with the trapezoidal rule.

### cb_crest_factor — peak relative to typical amplitude

Measures the largest absolute signal peak relative to RMS amplitude.

$$
\text{crest factor}=\frac{\max_i|x_i|}{\operatorname{RMS}(x)}
$$

Remove the fitted straight-line baseline from each entire component recording (linear detrending), then filter to **1–15 Hz**. Window: **P−5 to P+40 seconds**. Calculate per component, then take their median.

### cb_spectral_entropy — spread of frequency power

Measures how power is distributed across frequencies: lower values mean it is concentrated at fewer frequencies; higher values mean a more even spread. $Q_k$ is the spectral power value at frequency bin $k$, $K$ is the number of bins, and $p_k$ is its fraction of the total.

$$
p_k=\frac{Q_k}{\sum_{j=1}^{K}Q_j},
\qquad
\text{entropy}=-\frac{\sum_{k=1}^{K}p_k\ln\!\left(\max(p_k,10^{-15})\right)}{\ln K}
$$

Uses the same linear detrending and **1–15 Hz** filter as crest factor. Window: **P to P+40 seconds**. Welch segments: **4 seconds**; bins: **from 1 Hz up to but excluding 15 Hz**. Calculate per component, then take their median.
