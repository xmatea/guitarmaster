import numpy as np
import math
import librosa
import scipy.signal as signal
from matplotlib import pyplot as plt
from pathlib import Path
from io import BytesIO
import guitarpro as gp
import sys
import time
import pyaudio
import wave

AUDIO_SAMPLES = {path.stem: str(path) for path in (Path(__file__).parent / "test_recordings").glob("*.wav")}
SCORE_THRESHOLD = 0.05

def detect_note_times(data, sampling_frequency):
	# get parameters
	print(data)
	T = 1 / sampling_frequency
	N = len(data)
	t = N / sampling_frequency

	# fourier and half-rectifier
	f0, t0, Zxx = signal.stft(data, fs=sampling_frequency, nperseg=100, return_onesided=True)
	M = 1/len(Zxx[0])*np.sum(((np.real(Zxx) + np.abs(Zxx))/2), axis=0)
	
	# filter and get peaks
	b, a = signal.iirfilter(4, Wn=0.01, fs=0.5, btype="low", ftype="butter")
	m = signal.lfilter(b, a, M)
	peaks = signal.find_peaks(m, height=1e-7, prominence=0.1e-5)[0]

	return t0[peaks]

def tick_to_time(ticks: int, bpm: int) -> float:
	return 60 * (ticks/960)/bpm



def get_note_timings(song: gp.Song, track_num: int = 0) -> list[float]:
    tempo = song.tempo
    track = song.tracks[track_num]
    note_timings = []

    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if len(beat.notes) != 0 or all([note.type == gp.NoteType.rest for note in beat.notes]):
                    note_timings.append(
                        tick_to_time(beat.start, tempo))

    note_timings = sorted(set(note_timings))
    return note_timings

def calculate_score(x, ref, threshold):
	base = 100
	delta = abs(x-ref)

	if (delta >= 5*threshold):
		score = 0
	elif delta <= threshold:
		score = 100
	else:
		dev = delta/threshold
		score = 100-(dev*25)
	return score


def record():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1 if sys.platform == "darwin" else 2
	RATE = 44100
	RECORD_SECONDS = 5
	file = BytesIO()

	with wave.open(file, "wb") as wf:
		p = pyaudio.PyAudio()

		wf.setnchannels(CHANNELS)
		wf.setsampwidth(p.get_sample_size(FORMAT))
		wf.setframerate(RATE)

		stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

		print("Recording...")

		for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
			wf.writeframes(stream.read(CHUNK))
			print("Done")
			stream.close()
			p.terminate()
	return file




if __name__ == "__main__":
	song_name = "FreezingMoon"

	#user_input = record()
	data, sf = librosa.load(AUDIO_SAMPLES[song_name])
	song = gp.parse(f"gp5/{song_name}.gp5")
	
	played_times = detect_note_times(data, sf)
	print(f"{song.title} - {song.artist}")

	track = song.tracks[0]
	tempo = song.tempo
	timings = get_note_timings(song, 0)
	ix = len(played_times)

	played_times = np.array(played_times)
	actual_times = np.array(timings[:ix])

	print(f"Computer played: {actual_times}")
	print(f"You played: {played_times}")

	score = []
	for i in range(len(actual_times)):
		score.append(calculate_score(played_times[i], actual_times[i], SCORE_THRESHOLD))
		i += 1

	print(f"Your scores: {score}")


