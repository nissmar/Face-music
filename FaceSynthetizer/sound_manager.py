import wav_player
from time import time
import os
from random import sample
from math import ceil

# CONFIG
MINIMUM_TIME_BEFORE_RECORD_START = 1
NUMBER_OF_NOTES = 4

class SoundManager:
    modes = ['Piano', 'Drums', 'ElecGuitar','Orgue', 'Woodwinds']
    modes_sounds = [["",""] for _ in range(len(modes))]
    cur_mode = 0

    ready_for_record = False
    record_triggered_at = 0 # time when record key was pushed
    is_recording = False
    recording_begin = -1
    recorded_sounds = []

    def reset_soundtrack(self):
        self.measure_length = -1 # length of the measure in seconds. equal to -1 if still undefined
        self.measure_mode = -1 # instrument mode that defines measure
        self.soundtrack_last_start = 0 # stores the last time at which soundtrack began a new loop
        # array that stores soundtracks for each instrument mode, along with corresponding number of measures.
        # A number of measures of 0 means that the soundtrack for correspond mode is not defined yet
        self.modes_records = [[] for _ in range(len(self.modes))]
        self.soundtrack_length = -1 # stores the soundtrack length. -1 if soundtrack is empty

    def __init__(self):
        for mode in range(len(self.modes)):
            mode_name = self.modes[mode]
            sounds_files = list(filter(lambda x: x[-4:]=='.wav',os.listdir(f'notes/{mode_name}'))) #filter out non .wav formats
            sounds_files = [f'notes/{mode_name}/{x}' for x in sounds_files]
            self.modes_sounds[mode] = sounds_files[:NUMBER_OF_NOTES]
        self.reset_soundtrack()

    def echo_soundtrack(self):
        print('Current soundtrack:')
        for i in range(len(self.modes)):
            for j, record in enumerate(self.modes_records[i]):
                number_of_measures = record['number_of_measures']
                print(f'{self.modes[i]} {j}:', number_of_measures, 'measures')
        print('')

    def change_notes(self):
        """ change the notes of the current mode """
        mode = self.cur_mode
        mode_name = self.modes[mode]
        print('Changing notes for', mode_name, '!')
        sounds_files = list(map(lambda x: f'notes/{mode_name}/{x}.wav',os.listdir(f'notes/{mode_name}')))
        print(sounds_files)
        self.modes_sounds[mode] = sample(sounds_files, NUMBER_OF_NOTES)

    def get_mode(self):
        return self.modes[self.cur_mode]

    def change_mode(self, d=1):
        if not self.ready_for_record and not self.is_recording:
            self.cur_mode = (self.cur_mode+d)%(len(self.modes))
            return self.modes[self.cur_mode]

    def delete_cur_mode_last_record(self):
        if len(self.modes_records[self.cur_mode]) != 0: # no effect if no soundtrack for this mode
            print(f'Delete last record of {self.modes[self.cur_mode]}!')
            self.echo_soundtrack()
            self.modes_records[self.cur_mode].pop() # delete last soundtrack of current mode
            if len(self.modes_records[self.cur_mode]) == 0 and self.cur_mode == self.measure_mode:
                wav_player.mute_soundtrack()
                self.reset_soundtrack()
            self.build_full_soundtrack()

    def build_full_soundtrack(self):
        sounds = []
        length = -1
        records = [record for x in self.modes_records for record in x]
        if len(records) == 0:
            return
        # number of measures in full soundtrack
        measures_in_soundtrack = max([record['number_of_measures'] for record in records])
        for record in records:
            number_of_measures = record['number_of_measures']
            partial_soundtrack = record['soundtrack']
            record_sounds, record_length = partial_soundtrack['sounds'], partial_soundtrack['length']
            length = max(length, record_length)
            sounds += [(sound, offset + j*number_of_measures*self.measure_length) for j in range(measures_in_soundtrack//number_of_measures) for sound, offset in record_sounds]
        self.soundtrack_length = length
        full_soundtrack = {'sounds': sounds, 'length': length}
        if length > 0:
            wav_player.build_soundtrack_file(full_soundtrack)
            wav_player.play_soundtrack()
            self.soundtrack_last_start = time()

    def play_note(self, note_id):
        sound = self.modes_sounds[self.cur_mode][note_id]
        wav_player.play_note(sound)
        if self.is_recording:
            offset = time() - self.recording_begin
            self.recorded_sounds.append((sound,offset))

    def start_recording(self):
        self.ready_for_record = False
        self.is_recording = True
        self.recording_begin = time()
        self.recorded_sounds = []
        
    def end_recording(self):
        # ignore record if empty
        if self.recorded_sounds:
            record_length = time() - self.recording_begin
            # if measure is undefined, define it
            if self.measure_length == -1:
                self.measure_mode = self.cur_mode
                soundtrack_length = record_length
                self.measure_length = soundtrack_length
                number_of_measures = 1
            else:
                _, last_note_time = self.recorded_sounds[-1]
                number_of_measures = ceil(last_note_time/self.measure_length)
                soundtrack_length = number_of_measures * self.measure_length
            partial_soundtrack = {'sounds': self.recorded_sounds, 'length': soundtrack_length}
            self.modes_records[self.cur_mode].append({'number_of_measures': number_of_measures, 'soundtrack': partial_soundtrack})
            self.build_full_soundtrack()
        self.echo_soundtrack()
        self.is_recording = False

    def record_key(self):
        if self.is_recording:
            self.end_recording()
        elif not self.ready_for_record:
            self.ready_for_record = True
            self.record_triggered_at = time()

    def loop(self):
        new_soundtrack_loop = self.soundtrack_length > 0 and time() - self.soundtrack_last_start > self.soundtrack_length
        if new_soundtrack_loop:
            wav_player.play_soundtrack()
            self.soundtrack_last_start = time()
        if self.ready_for_record and (self.soundtrack_length == -1 or new_soundtrack_loop) and (time()-self.record_triggered_at) > MINIMUM_TIME_BEFORE_RECORD_START:
            self.start_recording()
        pos_in_measure = -1 # pos in measure in [0,1] for measure display
        if self.soundtrack_length > 0:
            pos_in_measure = ((time() - self.soundtrack_last_start) % self.measure_length)/self.measure_length
        number_of_records = len(self.modes_records[self.cur_mode]) # number of partial soundtracks for current mode
        return self.get_mode(), number_of_records, self.is_recording, self.ready_for_record, pos_in_measure

