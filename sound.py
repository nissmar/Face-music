import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

# PARAMS

SAMPLERATE = 44100 # may depend on the computer
FREQUENCY = 440
WAVE = []
start_idx = 0



# WAVE GENERATION

def sin(freq, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function
    """
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    return np.sin(2 * np.pi * freq * t)

def sin_fade(freq, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function
    """
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    return np.sin(2 * np.pi * freq * t)*np.exp(-(10000*t*t))

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
    outdata[:] =  sin(FREQUENCY, frames, start_idx)

    # frequency drop
    outdata[:] =  sin_drop(0.5, FREQUENCY, frames, start_idx)
    FREQUENCY -=0.5

    # chord
    # outdata[:] = chord([FREQUENCY,1.2*FREQUENCY,2*FREQUENCY] , frames, start_idx) 
    
    WAVE = outdata
    start_idx += frames

def show_wave(wave):
    plt.clf()
    plt.plot(wave)
    plt.draw()

def play2(freq):
    global FREQUENCY, WAVE
    WAVE = sin_fade(freq,1000)
    sd.stop()
    sd.play(WAVE)
    show_wave(WAVE)
    

def play(freq):
    global FREQUENCY, WAVE
    FREQUENCY = freq
    stream = sd.OutputStream(channels=1, callback=callback)
    stream.start()
    sd.sleep(800)
    stream.stop()
    show_wave(WAVE)

def onclick(event):
    print(event.x)
    play2(event.x)
    
# interaction
if __name__ == "__main__":
    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()