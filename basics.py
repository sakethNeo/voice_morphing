import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from librosa import lpc

# Load the audio file
audio_path = "C:/Users/ambal/Downloads/ee_moon_merged.mp3"
y, sr = librosa.load(audio_path, sr=None)
print(f"Sample rate: {sr} Hz")
print(f"Audio duration: {len(y)/sr:.2f} seconds")

# Set up parameters
frame_length = 2048
hop_length = 512
n_fft = 2048  # n_fft is typically frame_length for STFT-based features

# 1. WAVEFORM (Amplitude Envelope)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform (Amplitude Envelope)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.grid(True)
plt.show()

# 2. ZERO CROSSING RATE (ZCR)
zcr = librosa.feature.zero_crossing_rate(
    y, frame_length=frame_length, hop_length=hop_length
)
# Calculate time axis specifically for ZCR
zcr_time = librosa.frames_to_time(
    np.arange(zcr.shape[1]), sr=sr, hop_length=hop_length, n_fft=frame_length
)

plt.figure(figsize=(14, 4))
plt.plot(zcr_time, zcr[0])
plt.title("Zero Crossing Rate (ZCR)")
plt.ylabel("ZCR")
plt.xlabel("Time (s)")
plt.grid(True)
plt.show()


# 3. SHORT-TIME ENERGY (STE)
def compute_ste(signal, frame_length=2048, hop_length=512):
    energy = []
    # Adjusted to ensure full frames are processed, last frame might be partial if hop_length does not divide remaining signal
    for i in range(0, len(signal) - frame_length + 1, hop_length):
        frame = signal[i : i + frame_length]
        frame_energy = np.sum(frame**2)
        energy.append(frame_energy)
    return np.array(energy)


ste = compute_ste(y, frame_length, hop_length)
# STE is computed manually without librosa's centering, so its time calculation is independent
ste_time = np.arange(len(ste)) * hop_length / sr

plt.figure(figsize=(14, 4))
plt.plot(ste_time, ste)
plt.title("Short-Time Energy (STE)")
plt.ylabel("Energy")
plt.xlabel("Time (s)")
plt.grid(True)
plt.show()

# 4. FUNDAMENTAL FREQUENCY (F0)
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=frame_length,
)
# Calculate time axis specifically for F0
f0_time = librosa.frames_to_time(
    np.arange(len(f0)), sr=sr, hop_length=hop_length, n_fft=frame_length
)

plt.figure(figsize=(14, 4))
plt.plot(f0_time, f0, "o-", markersize=3)
plt.title("Fundamental Frequency (F0) Contour")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.grid(True)
plt.show()

# 5. INTENSITY (RMS Energy)
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
# Calculate time axis specifically for RMS
rms_time = librosa.frames_to_time(
    np.arange(rms.shape[1]), sr=sr, hop_length=hop_length, n_fft=frame_length
)

plt.figure(figsize=(14, 4))
plt.plot(rms_time, rms[0])
plt.title("Intensity (RMS Energy)")
plt.ylabel("RMS Energy")
plt.xlabel("Time (s)")
plt.grid(True)
plt.show()


# 6. PERIODICITY (using autocorrelation)
def autocorrelation(x, max_lag=1000):
    # Apply a Hamming window to reduce spectral leakage
    x = x * np.hamming(len(x))
    result = np.correlate(x, x, mode="full")
    mid = result.size // 2
    return result[mid : mid + max_lag]


# Take a representative frame for periodicity analysis
frame_duration = 0.03  # 30 milliseconds
frame_length = int(frame_duration * sr)

# --- 1. Analysis of the "ee" (Female Voice) ---
# Center the frame at t = 0.6 seconds
frame_start_female = int(0.6 * sr)
frame = y[frame_start_female : frame_start_female + frame_length]
autocorr = autocorrelation(frame)
lag_axis = np.arange(len(autocorr))

plt.figure(figsize=(14, 4))
plt.plot(lag_axis, autocorr)
plt.title("Periodicity (Autocorrelation Function)")
plt.ylabel("Autocorrelation")
plt.xlabel("Lag (samples)")
plt.grid(True)
plt.show()

# 7. SPECTRAL ENVELOPE (Spectrogram)
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
D_magnitude = np.abs(D)
D_db = librosa.amplitude_to_db(D_magnitude, ref=np.max)

plt.figure(figsize=(14, 6))
# librosa.display.specshow handles the time axis internally
librosa.display.specshow(
    D_db, sr=sr, hop_length=hop_length, y_axis="log", x_axis="time", cmap="viridis"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Spectral Envelope (Spectrogram)")
plt.show()


# 8. CEPSTRUM
def compute_cepstrum(frame):
    # Compute the power spectrum
    fft = np.fft.fft(frame)
    power_spectrum = np.abs(fft) ** 2
    # Take log and inverse FFT
    # It boosts the amplitude of lower-energy frequencies and reduces the range of the amplitudes,
    # making underlying periodic patterns in the spectrum more apparent.
    log_power_spectrum = np.log(power_spectrum + 1e-12)
    cepstrum = np.fft.ifft(log_power_spectrum).real
    return cepstrum


# Compute cepstrum for the same frame used in periodicity
cepstrum = compute_cepstrum(frame)
quefrency = np.arange(len(cepstrum)) / sr

plt.figure(figsize=(14, 4))
plt.plot(quefrency[: len(cepstrum) // 2], cepstrum[: len(cepstrum) // 2])
plt.title("Cepstrum")
plt.ylabel("Amplitude")
plt.xlabel("Quefrency (s)")
plt.grid(True)
plt.show()


start_index = 1

# It's also helpful to only plot up to a certain quefrency,
# as very high quefrencies are not relevant for pitch.
# A max quefrency of 0.02s corresponds to a pitch of 50 Hz, a good lower bound.
# We need to find the index that corresponds to 0.02s
end_index = int(0.02 * sr)


plt.figure(figsize=(14, 4))

# Plot the cepstrum, EXCLUDING the first coefficient
plt.plot(quefrency[start_index:end_index], cepstrum[start_index:end_index])

plt.title("Cepstrum (Pitch Analysis)")
plt.ylabel("Amplitude")
plt.xlabel("Quefrency (s)")
plt.grid(True)

# Find and mark the peak which corresponds to the pitch
# The relevant range for human pitch is roughly 70 Hz to 400 Hz
# which corresponds to quefrencies of ~0.0025s to 0.014s
quefrency_range_start = int(0.0025 * sr)
quefrency_range_end = int(0.014 * sr)

# Find the index of the maximum value in the relevant range
peak_index = np.argmax(cepstrum[quefrency_range_start:quefrency_range_end])
peak_index += quefrency_range_start  # Add the offset

peak_quefrency = quefrency[peak_index]
peak_amplitude = cepstrum[peak_index]

# Calculate the pitch in Hz
pitch_hz = 1 / peak_quefrency

# Add a marker to the plot
plt.axvline(
    x=peak_quefrency, color="r", linestyle="--", label=f"Pitch: {pitch_hz:.2f} Hz"
)
plt.legend()

plt.show()

print(f"The detected fundamental frequency (pitch) is {pitch_hz:.2f} Hz.")


# 9. MFCCs (Mel Frequency Cepstral Coefficients)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)

plt.figure(figsize=(14, 6))
# librosa.display.specshow handles the time axis internally
librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, x_axis="time")
plt.colorbar()
plt.ylabel("MFCC Coefficients")
plt.title("MFCCs (Mel Frequency Cepstral Coefficients)")
plt.show()

# 10. LPC SPECTRAL ENVELOPE FOR A SINGLE FRAME
# Define LPC order - a common heuristic is 2 + sr/1000
order = int(2 + sr / 1000)

# Extract a single frame from the middle of the audio
center_frame = np.int64(len(y) / (8 * hop_length))
start = center_frame * hop_length
end = start + frame_length
y_frame = y[start:end]

# Get LPC coefficients
a = librosa.lpc(frame, order=order)

# Get the frequency response of the LPC filter
w, h = signal.freqz(1, a, worN=n_fft)

# Get the spectrum of the original frame
frame_spectrum = np.fft.fft(frame * np.hanning(len(frame)), n_fft)
frame_power_spectrum = 20 * np.log10(np.abs(frame_spectrum))

freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)  # Frequencies for the positive spectrum

plt.figure(figsize=(14, 5))
plt.plot(
    freqs, frame_power_spectrum[: n_fft // 2 + 1], label="Original Spectrum"
)  # Spectrum
plt.plot(
    w * sr / (2 * np.pi),
    20 * np.log10(np.abs(h)),
    "r",
    linewidth=3,
    label="LPC Spectral Envelope",
)  # LPC
plt.title("LPC Spectral Envelope vs. Original Spectrum (Single Frame)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True)
plt.xlim(0, 5000)
plt.show()
