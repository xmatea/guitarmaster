import numpy as np
import librosa
from statsmodels import api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

note = "Cchord"
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
T = 1 / sampling_frequency
N = len(data)
t = N / sampling_frequency
f = sampling_frequency * np.arange((N/2)) / N; # frequencies
data_subset = data[80000:]
N_sub = len(data_subset)

print("--------------------------------x")
print(f"Note: {note}")
print(f"Length of recording: {t:.2f}s")
print(f"Number of samples {N}")
print(f"Sampling period {T:.3e}s")

"""
def animate(i):
	if (i > (N-10000))/100:
		 i = 0
	data_subset = data[i*100:]
	N_sub = len(data_subset)
	Y_k = np.fft.fft(data_subset)[:int(N_sub/2)]/N_sub # FFT
	Y_k[1:] = 2*Y_k[1:] # Single-sided spectrum
	Pxx = np.abs(Y_k) # Power spectrum
	line.set_ydata(Pxx[:5000])
	return line,

line, = ax.plot(f[:5000], Pxx[:5000], linewidth=2)

ani = animation.FuncAnimation(
fig, animate, interval=20, blit=True, save_count=50, repeat=True)
"""

Y_k = np.fft.fft(data)[:int(N/2)]/N # FFT
Y_k[1:] = 2*Y_k[1:] # Single-sided spectrum
Pxx = np.abs(Y_k) # Power spectrum

fig, ax = plt.subplots()
#plt.plot(f[:5000], Pxx[:5000], linewidth=2)
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
#plt.savefig(f"{note}".png)
#
#plot_acf(data, lags=100)
#plt.show()

auto = sm.tsa.acf(data_subset, nlags=2000)

peaks = np.array(find_peaks(auto)[0]) # Find peaks of the autocorrelation
#lag = peaks[2] # Choose the first peak as our pitch component lag

pitches = sampling_frequency / peaks # Transform lag into frequency
#print(f"\nExpected frequency: {FREQUENCIES[note]} Hz")
print(f"Detected pitches:\n1. {pitches[0]:.2f}Hz\n2. {pitches[1]:.2f}Hz\n3. {pitches[2]:.2f}Hz\n4. {pitches[3]:.2f}Hz\n5. {pitches[4]:.2f}Hz\n6. {pitches[5]:.2f}Hz\n")