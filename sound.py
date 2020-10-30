from synthesizer import Player, Synthesizer, Waveform
import matplotlib.pyplot as plt
import numpy as np

# Le synthé
player = Player()
player.open_stream()
synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)
chord = [440.0, 550.0, 660.0]
player.play_wave(synthesizer.generate_chord(chord, 0.5))

# interaction
fig, ax = plt.subplots()

def plot(x):
    plt.clf()
    plt.plot(np.sin(6*x/440*2*3.14*np.linspace(0,1,300)))
    plt.draw()
    
def onclick(event):
    player.play_wave(synthesizer.generate_constant_wave(abs(event.x), 0.1))
    plot(event.x)
    
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plot(440)

plt.show()