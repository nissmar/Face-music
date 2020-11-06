import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

# PARAMS

SAMPLERATE = 44100 # may depend on the computer
FREQUENCY_PLAYING = 440
FREQUENCY = 440
WAVE = []
start_idx = 0
phase = 0



# WAVE GENERATION

def sin(freq, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function
    """
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    return np.sin(2 * np.pi * freq * t)

    
def sin_transition(freq1, freq2, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function with sinusoidal transition
    """
    t = (start_idx + np.arange(frames)) / samplerate

    trans = (np.sin(np.pi/4*(np.linspace(0,1,frames)-0.5)*4)+1)/2 # store to not recompute each time
    return np.sin(2 * np.pi * (freq1 + (freq2-freq1)*trans) * t).reshape(-1,1)

  
def sin_transition2(freq1, freq2, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function with linear transition
    """
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    return np.sin(2 * np.pi * (freq1*(t[-1]-t) + freq2*(t-t[0]))/(t[-1]-t[0]) * t).reshape(-1,1)


def sin_transition3(freq1, freq2, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function with linear transition
    """
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    return np.sin(2 * np.pi * (freq1*(t[-1]-t)*(t[-1]-t) + freq2*(t-t[0])*(t-t[0]))/(t[-1]-t[0])/(t[-1]-t[0]) * t).reshape(-1,1)

def sin_transition4(freq1, freq2, frames, start_idx=0, samplerate=SAMPLERATE):
    """Regular sine function with linear transition
    """
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    phase = (freq2*(t-t[0])*(t-t[0]) - freq1*(t[-1]-t)*(t[-1]-t))/2/(t[-1]-t[0]) 
    return np.sin(2 * np.pi * phase).reshape(-1,1)
# SOUNDDEVICE
def callback(outdata, frames, time, status):
    print(frames)
    global start_idx, FREQUENCY,FREQUENCY_PLAYING, WAVE

    # standard sine
    if (FREQUENCY_PLAYING != FREQUENCY):
        WAVE = sin_transition(FREQUENCY_PLAYING, FREQUENCY, frames, start_idx)
        FREQUENCY_PLAYING += 1
    else:
        WAVE = sin(FREQUENCY_PLAYING, frames, start_idx)
    outdata[:] =  WAVE
    start_idx += frames


# stream = sd.OutputStream(channels=1, callback=callback,blocksize=0)
# stream.start()
# i=0
# # input()
# while not(rave(i)):
#     i-=1
#     if (i%1000000) == 0:
#         FREQUENCY += 10
#     # print(i)
# stream.stop()


frames = 512
L=[]
func = sin_transition4
f1 = 100
f2 =800
L.append(func(f1,f1,frames))
L.append(func(f1,f2,frames,frames))
L.append(func(f2,f2,frames,2*frames))

plt.plot(np.concatenate(L),'-')
plt.show()
