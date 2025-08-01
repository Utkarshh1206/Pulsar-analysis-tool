import streamlit as st
import base64

st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px;'>
        🔭 Pulsar Analyzer
    </h1>
    <h4 style='text-align: center; color: #c5c6c7;'>
        Analyze Dual-Channel Voltage Data with FFT, Dynamic SPectrum, and More
    </h4>
    <hr style="border: 1px solid #45a29e;">
    """,
    unsafe_allow_html=True
)

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

import base64

def set_background(image_file):
    with open(image_file, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("nasa-Yj1M5riCKk4-unsplash.jpg")  


st.set_page_config(page_title="Pulse Simulator", layout="centered")
st.title("🔭 Pulse Simulator — Interactive Radio Analysis")

#uploaded_file = st.file_uploader("📁 Upload your Vela data (.txt)", type=["txt", "dat", "csv"])
with st.spinner("Processing data... This may take a while."):
    time.sleep(2)  
file_type = st.radio("Select file type", ["Text/CSV file", "Spec file"])

uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv", "spec"])

voltage_data=None

if uploaded_file is not None:
    if file_type == "Text/CSV file":
        try:
            voltage_data = np.loadtxt(uploaded_file)
            st.success("Text/CSV data loaded successfully!")
        except:
            st.error(f"Error reading text/csv file")
        
    elif file_type == "Spec file":
        try:
            raw_bytes = uploaded_file.read()
            raw_data = np.frombuffer(raw_bytes, dtype=np.float64)
            if raw_data.size % 2 != 0:
                st.warning("Spec file data is not perfectly divisible by 2 — trimming last element.")
                raw_data = raw_data[:-1]  # trim last to make it even
            reshaped_data = raw_data.reshape(-1, 2)
            voltage_data = reshaped_data
            st.success("Spec binary file loaded and reshaped!")
        except Exception as e:
            st.error(f"Error reading spec file :{e}")


st.subheader(f'length of data :{len(voltage_data)}')


sampling_rate = st.number_input("⏱️ Sampling chunks", value=0.0)
bandwidth = st.number_input("📶 Bandwidth (in Hz)", value=0.0)

if voltage_data.ndim < 2 or voltage_data.shape[1] < 2:
     st.error("Expected 2 columns (North & South). Check your file format.")
else:        
     north_channel = voltage_data[:, 0]
     south_channel = voltage_data[:, 1]

channel = st.radio("🎚️ Select Channel for Analysis", ["North", "South"])
signal = north_channel if channel == "North" else south_channel

st.subheader("📉 Voltage-Time Waveform for Both Channels")
fig, ax = plt.subplots()
ax.plot(north_channel, label="Northern Channel", color="blue")
ax.plot(south_channel, label="Southern Channel", color="yellow")
ax.set_xlabel("Time")
ax.set_ylabel("Voltage")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader(f"📊 Histogram of Raw Voltage — {channel} Channel")
fig_hist = plt.figure(figsize=(8, 3))

mean = np.mean(signal)
std = np.std(signal)

count,bins,_= plt.hist(signal, bins=80, color='yellow', edgecolor='black', label='Observed Histogram')

from scipy.stats import norm

x = np.linspace(min(bins), max(bins), 1000)
pdf = norm.pdf(x, mean, std)

bin_width = bins[1] - bins[0]
pdf_scaled = pdf * len(signal) * bin_width
plt.plot(x, pdf_scaled, 'r--', linewidth=2, label='Theoretical Gaussian PDF')

plt.xlabel('Voltage signal')
plt.ylabel('No. of occurrences')
plt.title('Distribution of voltage signal for northern field')
plt.legend()
plt.tight_layout()
plt.title("Voltage Distribution Histogram")
st.pyplot(fig_hist)


st.subheader(f'Mean of the data : {mean}')
st.subheader(f'standard deviation: {std}')

from scipy.stats import expon
st.subheader(f"Power distribution for - {channel} Channel")
pow = signal**2
loc, scale = expon.fit(pow)

fig_pow, ax = plt.subplots(figsize=(8, 3))

count, bins, _ = ax.hist(
    pow, bins=50, edgecolor='black', label='Observed',
    alpha=0.7, color='blue', density=False
)

x = np.linspace(min(bins), max(bins), 1000)
pdf = expon.pdf(x, loc, scale)
bin_width = bins[1] - bins[0]
pdf_scaled = pdf * len(pow) * bin_width
ax.plot(x, pdf_scaled, 'r--', label='Theoretical')

ax.set_title('Power distribution for Northern channel')
ax.set_xlabel('Intensity value')
ax.set_ylabel('No. of occurrences')
ax.legend()

st.subheader(f"Power distribution for - {channel} Channel")
st.pyplot(fig_pow)

st.subheader(f'Fourier ransform of - {channel} channel')
fourier_a = np.fft.fft(signal)
fig_fourier = plt.figure(figsize=(8,3))
plt.plot(fourier_a)
plt.title(f'Fourier ransform of - {channel} channel')
plt.grid(True)
st.pyplot(fig_fourier)

Nf = sampling_rate
fmax = bandwidth
st.subheader(f'Sampling : {Nf}')
st.subheader(f'bandwidth : {fmax} Hz')
df = fmax/Nf
st.subheader(f'frequency resolution : {df*1e-3} KHz')
tmax = 1/df
st.subheader(f'time resolution : {tmax} s')
dt = 1/(2*fmax)
st.subheader(f'sampling interval : {dt} s')
Nfft = len(signal)*tmax
st.subheader(f'number of points : {Nfft}')
Nspec = len(signal)//Nfft
st.subheader(f'Number of spectral points : {Nspec}')
time = Nfft*dt
st.subheader(f'Time resolution of spectra: {time}')

st.subheader(f"⚡ Averaged Power Spectrum — {channel} Channel")
segment_size = st.number_input("🔁 Number of FFT slices to average", min_value=1, max_value=1024, value=50, step=1)
num_segments = len(signal) // segment_size

if num_segments < 1:
    st.warning("⚠️ Not enough data points for 512-sample segmenting.")
else:
    c = []
    g = []

    for i in range(1, num_segments):
        a_i = signal[segment_size * i : segment_size * (i + 1)]
        fft_vals = np.fft.fft(a_i)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.fftfreq(len(a_i), d=1/sampling_rate)

        half_n = len(freqs) // 2
        c.append(power[1:half_n])
        g.append(freqs[1:half_n])

    mean_power = np.mean(c, axis=0)
    mean_freq = np.mean(g, axis=0)

    fig_avg = plt.figure(figsize=(8, 3))
    plt.plot(mean_freq, mean_power, color='limegreen')
    plt.title(f"Mean Power Spectrum (Segment size = 512)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.grid(True)
    st.pyplot(fig_avg)

st.subheader(f'Dynamic spectrum - {channel} Channel')

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
total_points = int(Nfft) * int(Nspec)
a1 = np.array(signal[:total_points])
a1_reshaped = a1.reshape((int(Nspec), int(Nfft)))
p_a = np.fft.rfft(a1_reshaped, axis=1) 
p_abs = np.abs(p_a)**2  
def plot_dynamic_spectrum(signal, Nfft, Nspec, channel, dt, fmax, cmap='RdGy'):

    if len(signal) < total_points:
        st.error(f"Signal too short: Expected {total_points} points, got {len(signal)}")
        return

    st.markdown("### Contrast Scaling")
    if st.checkbox("🎚️ Customize contrast range"):
        low = st.slider("Min Percentile", 0, 50, 5)
        high = st.slider("Max Percentile", 50, 100, 99)
    else:
        low, high = 5, 99

    vmin = np.percentile(p_abs, low)
    vmax = np.percentile(p_abs, high)

    use_log = not st.checkbox("📈 Use Linear Scale (Uncheck for Log)", value=False)
    power_plot = np.log10(p_abs.T + 1e-12) if use_log else p_abs.T


    N_freqs = int(Nfft // 2 + 1)
    freq_axis = np.linspace(0, fmax, N_freqs) / 1e6

    total_duration = Nspec * dt * 2 * Nfft
    time_axis = np.linspace(0, total_duration, Nspec)

    # --- Step 6: Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        power_plot,
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"Dynamic Spectrum – {channel} Channel")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [MHz]")
    ax.grid(False)
    plt.colorbar(im, ax=ax, label="Log Power" if use_log else "Power")
    st.pyplot(fig)

plot_dynamic_spectrum(signal=signal,
                      Nfft=int(Nfft),
                      Nspec=int(Nspec),
                      channel=channel,
                      dt=dt,       
                      fmax=fmax,    
                      cmap='inferno')
dedis = st.checkbox("Apply dedispersion")
if dedis:
    DM = st.number_input("Enter Dispersion Measure (pc/cm³):", value=0.0)
    st.subheader(f'Dedispersion of - {channel} Channel')
    with st.expander("🔧 Advanced Dedispersion Settings"):
        f_high = st.number_input("Upper Frequency (Hz)", value=0.0, format="%i")
        f_low = st.number_input("Lower Frequency (Hz)", value=0.0, format="%i")
        s1 = np.transpose(np.square(np.abs(p_a)))
        freq_bins = s1.shape[0]
        fr = np.linspace(f_low, f_high, freq_bins)

        if f_high > f_low and f_low != 0:
            td = 4.15e6 * DM * ((1 / f_low**2) - (1 / f_high**2))  # in milliseconds
            td_sec = td / 1000  # in seconds
            st.success(f"Dispersion Delay (Δt): {td:.2f} ms ({td_sec:.4f} s)")
        else:
            st.error("Make sure 'Upper Frequency' is greater than 'Lower Frequency' and not zero.")

        tshift =np.round((td)/(Nfft*dt))
        tshift=tshift.astype(int)
        s2_dedisp = np.transpose(np.square(np.abs(p_a)))

        for k, row in enumerate(s2_dedisp):
            s2_dedisp[k, :] = np.roll(s2_dedisp[k, :], -tshift)

        fig_dedis = plt.figure(figsize=(8,3))
        plt.imshow(s2_dedisp, aspect='auto', extent = [0, voltage_data.shape[0] * Nfft * dt * 1000, f_low, f_high], cmap='copper')
        plt.title(' Dedispersion of southern channel')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency Channel')
        plt.colorbar()
        plt.grid(False)
        st.pyplot(fig_dedis)

st.subheader(f'dedispered signal for {channel}')
t1 = np.average(s2_dedisp,axis=0)
fig_disp_signal = plt.figure(figsize=(10,5))
plt.plot(t1)
plt.title('dedispersed signal for northern')
plt.xlabel('time')
plt.ylabel('Power')
t1 = np.roll(t1,100)
st.pyplot(fig_disp_signal)

st.subheader('Pulse Profile')

from scipy.signal import correlate
from scipy.signal import find_peaks


def estimate_period(t1,dt,fallback_period = 0.089):
    from scipy.signal import correlate
    corr = correlate(t1,t1,mode='full')
    corr = corr[len(corr)//2:]

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(corr, distance=50)

    if len(peaks) > 1:
        lag_diffs = np.diff(peaks)
        avg_lag = np.mean(lag_diffs)
        estimated_period = avg_lag * dt  # Convert lag to time
    else:
        estimated_period = fallback_period

    return estimated_period


know_period = st.checkbox("Do you know the pulsar period (tperiod)?")
if know_period:
    tp = st.number_input('enter the time period(in s):',value = 0.000000)

else:
    tperiod = estimate_period(t1, dt)
    st.success(f"Estimated pulse period: {tperiod:.6f} s")


fold_size = int(tp/(Nfft*dt))
bins = len(t1)//fold_size
print(fold_size,bins)

pulse_profile = np.zeros(fold_size)

for k in range(fold_size):
  for j in range(bins):
    pulse_profile[k] += t1[k+j*fold_size]

x=np.linspace(0,len(pulse_profile),len(pulse_profile),len(pulse_profile)*2)
mu = 350
sigma = 67
y= np.exp(-(x-mu)**2/(2*sigma**2))*1.4e6/(np.sqrt(2*np.pi)*sigma)
fig_pulse = plt.figure(figsize=(7, 6))
plt.title('Dedispersed Pulse Profile')
plt.xlabel('Time')
plt.ylabel('Signal Power')
plt.plot(pulse_profile, label='Observed power series')
plt.legend()
st.pyplot(fig_pulse)

st.subheader(f'rms correction')
mu_p = np.mean(pulse_profile)
rms_error = np.sqrt(np.sum(pulse_profile**2-mu_p**2)/len(pulse_profile))
rms = pulse_profile - rms_error
fig_rms = plt.figure(figsize=(7, 6))
plt.title('Dedispersed Pulse Profile')
plt.xlabel('Time')
plt.ylabel('Signal Power')
plt.plot(rms, label='Observed power series')
st.pyplot(fig_rms)