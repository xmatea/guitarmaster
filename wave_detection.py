import numpy as np
import math
import librosa
import scipy.signal as signal
from matplotlib import pyplot as plt
from pathlib import Path

note = "FadeToBlack"
AUDIO_SAMPLES = {path.stem: str(path) for path in (Path(__file__).parent / "test_recordings").glob("*.wav")}
FREQUENCIES = {
"A2": 110,
"B3": 247,
"D3": 147,
"E2": 82,
"E3": 165,
"E4": 330,
"G3": 196
}

data, sampling_frequency = librosa.load(AUDIO_SAMPLES[note])

def detect_note_times(data, sampling_frequency):
	# get parameters
	T = 1 / sampling_frequency
	N = len(data)
	t = N / sampling_frequency

	# fourier and rectifier
	f0, t0, Zxx = signal.stft(data, fs=sampling_frequency, nperseg=100, return_onesided=True)
	M = 1/len(Zxx[0])*np.sum(((np.real(Zxx) + np.abs(Zxx))/2), axis=0)
	
	# filter and get peaks
	b, a = signal.iirfilter(4, Wn=0.01, fs=0.5, btype="low", ftype="butter")
	m = signal.lfilter(b, a, M)
	peaks = signal.find_peaks(m, height=1e-7, prominence=0.1e-5)[0]

	return t0[peaks]

T = 1 / sampling_frequency
N = len(data)
t = N / sampling_frequency
t_ms = t*1000
time = np.linspace(0, t_ms, N)
frame = 50

fig, ax = plt.subplots()
plt.plot(time, data)
plt.title("Raw data")
#plt.show()

f0, t0, Zxx = signal.stft(data, fs=sampling_frequency, nperseg=100, return_onesided=True)
N_2 = len(Zxx[0])
t_2 = np.linspace(0, t, len(Zxx[0]))


M = 1/len(Zxx[0])*np.sum(((np.real(Zxx) + np.abs(Zxx))/2), axis=0)
b, a = signal.iirfilter(4, Wn=0.01, fs=0.5, btype="low", ftype="butter")
m = signal.lfilter(b, a, M)

plt.title("Smoothed curve")
plt.plot(t_2, m)
#plt.show()

plt.pcolormesh(t0, f0, np.abs(Zxx), vmin=0, shading='gouraud')
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()
"""
"""
peaks = signal.find_peaks(m, height=1e-7, prominence=0.1e-5)[0]
freqs = m[peaks]
times = t0[peaks]

plt.plot(t_2, m)
plt.title("Detected peaks")
plt.scatter(times, freqs, marker="x", color="#00aaff")
#plt.show()
#note_start = t0[np.unravel_index(np.argmax(Zxx), Zxx.shape)[1]]
"""

# print stuff for added excitement
print("--------------------------------x")
print(f"Note: {note}")
print(f"Length of recording: {t:.2f}s")
print(f"Number of samples {N}")
print(f"Sampling period {T:.3e}s")
print(f"\nTime at peak: {times}s")

"""

