import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

# PARAMS

SAMPLERATE = 44100 # may depend on the computer
FREQUENCY_PLAYING = 440 # current frequency
FREQUENCY = 440 # goal
WAVE = []
PHASE = 0 # phase



# WAVE GENERATION

def sin_transition(freq1, freq2, frames, samplerate=SAMPLERATE):
    """sine function with variable frequency"""
    global PHASE

    t = np.arange(frames)/ samplerate
    G = (freq2*t*t - freq1*(t[-1]-t)*(t[-1]-t))/2/t[-1] + t[-1]/2*freq1 # integrates (1-t)f1 + t*f2
    out = np.sin(2 * np.pi * G + PHASE).reshape(-1,1)    
    PHASE += 2 * np.pi * (G[-1] + freq2*t[1]) # phase shift
    return out

  

# SOUNDDEVICE
def callback(outdata, frames, time, status):
    global FREQUENCY,FREQUENCY_PLAYING, WAVE

    # standard sine
    WAVE = sin_transition(FREQUENCY_PLAYING, FREQUENCY, frames)
    outdata[:] =  WAVE

    FREQUENCY_PLAYING = FREQUENCY



def onclick(event):
    global FREQUENCY
    FREQUENCY = event.x
    plt.clf()
    plt.plot(WAVE)
    plt.draw()
    
def set_freq(freq):
    global FREQUENCY
    FREQUENCY = freq

# interaction
if __name__ == "__main__":
    stream = sd.OutputStream(channels=1, callback=callback,blocksize=10000)
    stream.start()

    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    stream.stop()

# frames = 512
# L=[]
# func = sin_transition
# f1 = 800
# f2 =1600
# L.append(func(f1,f1, frames))
# L.append(func(f1,f1, frames))
# L.append(func(f1,f2,frames))
# L.append(func(f2,f2,frames))
# L.append(func(f2,f2,frames))
# L.append(func(f2,f2,frames))
# plt.plot(np.concatenate(L),'-')
# plt.show()
