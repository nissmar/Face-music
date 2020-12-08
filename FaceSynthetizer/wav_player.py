from pydub import AudioSegment
from pygame import mixer
from numpy.random import rand

# we define the type 'soundtrack' as a dict with a list of tuple (second, sound file name) corresponding to the 'sounds' key representing the sounds
# of the soundtrack and the time they are played at, and a float corresponding to the 'length' key representing thesoundtrack length in seconds

soundtrack_file = 'loop.wav'

mixer.init()
channel_note = mixer.Channel(0)
channel_soundtrack = mixer.Channel(1)

def build_soundtrack_file(soundtrack):
    delayed = [AudioSegment.silent(duration=1000*offset) + AudioSegment.from_wav(file_name) for file_name,offset in soundtrack['sounds']]
    normalized = [AudioSegment.silent(duration=1000*soundtrack['length']).overlay(sound) for sound in delayed]
    resulting_sound = normalized[0]
    for sound in normalized[1:]:
        resulting_sound = resulting_sound.overlay(sound)
    resulting_sound.export(soundtrack_file, format = 'wav')


def play_note(file):
    sound = mixer.Sound(file)
    channel_note.play(sound)

def play_soundtrack():
    soundtrack = mixer.Sound(soundtrack_file)
    channel_soundtrack.play(soundtrack)

def mute_soundtrack():
    channel_soundtrack.pause()