import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis


import pandas as pd
import matplotlib.pyplot as plt
import h5py 




##### Basic Signal Processing

#Each waveform has three channels. E is the east-west direction, N is north-south, Z is vertical. Noe that we only ever care about N^2+E^2, so it makes no difference if these are rotated wrt to the labels.
CH = {"E": 0, "N": 1, "Z": 2}    

#Define a basic bandpass filter
def bandpass(x, fs, lo, hi, order=4): 
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)

#preprocessing function that substracts the mean and tapers the signal
def preprocess(tr3, fs):
    x = tr3 - tr3.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    taper = np.ones(n); k = int(0.05 * n)
    taper[:k] = np.hanning(2*k)[:k]; taper[-k:] = np.hanning(2*k)[k:]
    return x * taper

#computes the RMS of the waveform in a given window 
def window_rms(x, start, length): 
    start = max(int(start), 0)
    seg = x[..., start:start + int(length)]
    return np.sqrt(np.mean(seg**2, axis=-1)) if seg.shape[-1] > 0 else np.nan

#####







# First and Second feautre: Amplitude features
# log(P_z/S_h) and log(horizontal/vertical) at the S-wave

def amp_features(tr3, fs, p_samp, s_samp, lo=4, hi=8, p_len_s=1.5, s_len_s=3.0, noise_len_s=3.0, guard_s=0.5):
    tr3 = tr3.astype(float)          # Turn entries into floats
    n_dead = int((np.abs(tr3).max(axis=1) == 0).sum())   # count number of deads by measuring absolute value
    out = dict(log_ps=np.nan, log_hv_s=np.nan, snr_p=np.nan, snr_s=np.nan, n_dead=n_dead) #return nan if dead channels
    if np.isnan(s_samp) or n_dead > 0: #if no s or dead channels, return nan
        return out

    x = bandpass(preprocess(tr3, fs), fs, lo, hi) #filter signal
    p, s0 = int(p_samp), int(s_samp) #make arrivel time indices integers
    p_len = min(int(p_len_s * fs), s0 - p) # define length of p window as standard or shorter if S is too close. 
    if p_len < int(0.3 * fs):  #If P and S too close, Kill
        return out

    z = x[CH["Z"]]  #vertical channel.
    h = np.sqrt(x[CH["N"]]**2 + x[CH["E"]]**2)  #horizontal channel. 
    s_len, guard = int(s_len_s * fs), int(guard_s * fs)  #turn seconds into indices

    noise = window_rms(z, p - guard - int(noise_len_s * fs), int(noise_len_s * fs))   #noise measure
    P   = window_rms(z, p, p_len) # P z rms
    S_h = window_rms(h, s0, s_len) # S h rms
    S_z = window_rms(z, s0, s_len) # S z rms

    #return amplitude ratios 
    out.update(
        log_ps=np.log10(P / S_h),
        log_hv_s=np.log10(S_h / S_z),
        snr_p=np.log10(P / noise),
        snr_s=np.log10(S_h / noise),
    )
    return out



# Third and fourth feature: envelope features
 #rise_time and kurtosis

 def envelope_features(tr3, fs, p_samp, s_samp, lo=2, hi=10,pre_s=1.0, post_s=10.0, smooth_s=0.2):
    
    tr3 = tr3.astype(float)                           #change entries to float
    
    out = dict(rise_time=np.nan, env_kurtosis=np.nan) # the return if no S wave or dead Z channel
    if np.isnan(s_samp):                              #default if no S wave registered
        return out
    if np.abs(tr3[CH["Z"]]).max() == 0:             #return default if not Z channel
        return out

    x = bandpass(preprocess(tr3, fs), fs, lo, hi)  #filter all channels
    env = np.sqrt((x**2).sum(axis=0))       #envelope is old name, we are really just taking the 3d amplitude
    k = max(int(smooth_s * fs), 1)          #Take the max of the smooth_s (in ticks) variable and 1. 
    env = np.convolve(env, np.ones(k) / k, mode="same") #smooth envelope with a convolution

    a = max(int(p_samp - pre_s * fs), 0)  #a is the point at which we start the analysis. p arrival minus some value pre_s
    b = min(int(s_samp + post_s * fs), len(env)) #b is the point at which end the analysis
    e = env[a:b]  #slice the envelop
    if len(e) < fs:  #default if we don't have at leas 1 second
        return out

    # rise time: 10% -> 90% of max, on the way up to the peak
    i_peak = e.argmax() #find the highest peak (usually P)
    rise = e[:i_peak + 1] #slice into the portion before
    i10 = np.argmax(rise >= 0.1 * e.max()) #returns the first point at which e>= 0.1 of peak
    i90 = np.argmax(rise >= 0.9 * e.max()) #returns the first point at which e>= 0.9 of peak
    out["rise_time"] = (i90 - i10) / fs #record the time between these two points

    # excess kurtosis of the envelope over the event window
    out["env_kurtosis"] = kurtosis(e) #record the kurtosis of the envelop. 
    return out




#### Fifth feature: spectral energy ratio (low fq/high fq)

def spectral_features(tr3, fs, p_samp, pre_s=1.0, win_s=20.0,low=(1, 5), high=(5, 15)):  #figure out if these frequency intervals are reasonabe

    tr3 = tr3.astype(float) 
    
    out = dict(log_lowhigh=np.nan)          #Set default ouput to nan
    dead = np.abs(tr3).max(axis=1) == 0     #check for dead channels
    if dead.any():                          # return nan if channels are dead     
        return out
    

    x = preprocess(tr3, fs)       #preprocess the signal. #Important to not smooth it to not delete high frequencies etc.
    a = max(int(p_samp - pre_s * fs), 0) # beginning of the window
    b = min(int(p_samp + win_s * fs), x.shape[1]) # end of the window
    seg = x[:, a:b]                               # slice each channel 
    if seg.shape[1] < 2 * fs:                      #return default if window is shorter than 2sec
        return out

    f, pxx = welch(seg, fs=fs, nperseg=int(2 * fs), axis=-1) #returns the fourier tranfsorm f (with Welch method) and pxx is the power (squared frequencies)
    pxx = pxx.sum(axis=0)                                    #sum square powers

    m_low  = (f >= low[0])  & (f < low[1]) #true/false list or array of where the frequencies are in the low range
    m_high = (f >= high[0]) & (f < high[1]) #ditto for high range
    e_low  = np.trapezoid(pxx[m_low],  f[m_low]) # integrate power with trapezoid rule in the low frequency range
    e_high = np.trapezoid(pxx[m_high], f[m_high]) #integrate power in high frequency range. 
    out["log_lowhigh"] = np.log10(e_low / e_high) #we record the log(low/high) ratio
    return out
