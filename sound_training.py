import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

# PARAMS

SAMPLERATE = 44100 # may depend on the computer
FREQUENCY = 600
WAVE = []
start_idx = 0



# WAVE GENERATION

def sin(freq, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function
    """
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    return np.sin(2 * np.pi * freq * t)

def sin_drop(rate, freq, frames, start_idx=0, samplerate=SAMPLERATE):
    """Sine with linear frequency dropping
    """
    t = (start_idx + np.arange(frames)) / SAMPLERATE
    t = t.reshape(-1, 1)
    freq = freq - rate*(t-t[0])/(t[-1]-t[0]) # frequency drop
    return np.sin(2 * np.pi * freq * t)

def chord(freqs, frames, start_idx=0, samplerate=SAMPLERATE):
    """sum of sines
    """
    n = len(freqs)
    return sum([sin(freqs[i], frames, start_idx) for i in range(n)])/n

# SOUNDDEVICE
def callback(outdata, frames, time, status):
    global start_idx, FREQUENCY, WAVE

    # standard sine
    outdata[:] =  0.2*sin(FREQUENCY, frames, start_idx)

    WAVE = outdata
    start_idx += frames


def play():
    stream = sd.OutputStream(channels=1, callback=callback)
    stream.start()
    sd.sleep(800)
    stream.stop()


