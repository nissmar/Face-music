import sounddevice as sd
import soundfile as sf

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import gamma

# PARAMS

FREQUENCY = 440 # goal
SAMPLERATE = 44100 # may depend on the computer



def adjust(variable, target, d=20):
    """ variable <- target with a step of at most d """
    if variable>d+target:
        variable -= d
    elif variable<target-d:
        variable += d
    else:
        variable = target
    return variable

# WAVE GENERATION

def sin_transition(freq1, freq2, frames, samplerate=SAMPLERATE):
    """sine function with variable frequency"""
    t = np.arange(frames)/ samplerate
    G = (freq2*t*t - freq1*(t[-1]-t)*(t[-1]-t))/2/t[-1] + t[-1]/2*freq1 # integrates (1-t)f1 + t*f2
    return np.sin(2 * np.pi * G)

def sin_n(freq, frames, samplerate=SAMPLERATE):
    """sine function with phase"""
    t = np.arange(frames)/ samplerate
    return np.sin(2 * np.pi * freq * t)

def square(freq, frames, samplerate=SAMPLERATE):
    t = np.arange(frames)/ samplerate
    return 2*( ((t%(1.0/freq))<(1.0/freq)/2) - 0.5)

def triangle(freq, frames, samplerate=SAMPLERATE):
    t = np.arange(frames)/ samplerate
    return 2*( np.abs((signal.sawtooth(2*np.pi * freq * t-np.pi/2))) - 0.5 )

def envelope(x, frames, samplerate=SAMPLERATE):
    t = np.arange(frames)/ samplerate
    return np.sin((t/t.max())**x*np.pi)

# GOOD VIBES
def good1(freq,frames,samplerate=SAMPLERATE):
    env = envelope(1.4,frames)**2
    env2 = envelope(0.2,frames)
    wave = sin_n(freq,frames)
    wave2 = triangle(freq,frames)/2
    return env*wave +env2*wave2

def SineTri(freq,frames,samplerate=SAMPLERATE):
    return good1(freq,frames)+good1(freq*1.5,frames)+good1(freq*2,frames)/3+good1(2*freq*1.5,frames) 


def good2(freq,frames,samplerate=SAMPLERATE):
    env = envelope(1,frames)**2*sin_n(3.0/frames*SAMPLERATE,frames)
    wave = sin_n(freq,frames)
    wave2 = square(freq,frames)/2
    return env*(wave+wave2)


def Osc(freq,frames,samplerate=SAMPLERATE):
    return good2(freq,frames)+good2(freq*1.5,frames)+good2(freq*2,frames)/3+good2(2*freq*1.5,frames) 



def ModernChurch(freq,frames,samplerate=SAMPLERATE):
    env = envelope(1,frames)**0.5
    wave = sin_n(freq,frames)+2*sin_n(freq*2,frames)+2*sin_n(freq*3,frames)+10*triangle(freq*4,frames)+sin_n(freq*5,frames)
    return env*wave


def CryBaby(freq,frames,samplerate=SAMPLERATE):
    env = envelope(1,frames)**0.5
    f1 = freq*(0.9 +sin_n(0.6,frames)/3.0)
    f2 = freq*(0.9 +sin_n(0.3,frames)/3.0)
    wave = sin_n(f1,frames)+sin_n(2*f2,frames)+sin_n(3*f1,frames)
    return env*wave

def kick(freq,frames,x=0.1,samplerate=SAMPLERATE):
    return sin_n(freq,frames)*envelope(x,frames)**10

def bongo(freq,frames,samplerate=SAMPLERATE):
    x=0.08
    return kick(freq,frames,x)+kick(freq+40,frames,x)+kick(freq+80,frames,x)/4+kick(freq,frames,0.2)

def bell(freq,frames,samplerate=SAMPLERATE):
    x=0.08
    base = kick(freq,frames,x)+kick(freq+40,frames,x)
    env = envelope(0.3,frames)
    high = sin_n(2*freq,frames)*env**3+sin_n(3*freq,frames)*env+sin_n(4*freq,frames)*env**0.5
    return base+high


def rec(func,frames,freqs):
    i=0
    for f in freqs:
        wave = func(f,frames)
        wave = wave/max(wave)
        sf.write('output'+str(i)+'.wav', wave, SAMPLERATE)
        i+=1

INDEX=0
def onclick(event):
    global INDEX

    frames = 20000
    x = min(16*event.x/frames,2)
    y = min(event.y/800,1)
    freqs=[293.66, 392,493.88,587.33] 
    # freqs=[98,123.47,146.83,174.61]
    # freqs=[32.7,41.20,49,65.41]
    rec(bell,frames,[f for f in freqs])
    wave=Osc(freqs[INDEX]*1.5,frames)
    wave=wave/max(wave)
    # sf.write('output'+str(0)+'.wav', wave, SAMPLERATE)
    plt.clf()
    plt.plot(wave)
    plt.draw()
    sd.play(wave)
    # sf.write('output'+str(i)+'.wav', wave, SAMPLERATE)
    INDEX+=1
    INDEX=INDEX%4
 
# interaction
if __name__ == "__main__":
    # fig, ax = plt.subplots()
    # cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()

    frames = 200000
    freq = 400
    env = envelope(1.4,frames)**2
    env2 = envelope(0.2,frames)
    wave = sin_n(freq,frames)
    wave2 = triangle(freq,frames)/2
    # envelopes
    line1 = plt.plot(envelope(0.3,frames), label="x=0.3, n=1")
    # plt.legend(handles=line1)

    line2 =plt.plot(envelope(1.4,frames), label="x=1.4, n=1")
    # plt.legend(handles=line2)

    line3 =plt.plot(envelope(1.4,frames)**5, label="x=1.4, n=5")
    plt.legend(loc='lower right')

    # waves
    # plt.plot(sin_n(freq,frames))
    # plt.plot(triangle(freq,frames))
    # plt.plot(square(freq,frames))

    # example

    # plt.plot(Osc(200,frames))
    plt.show()
