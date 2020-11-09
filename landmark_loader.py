import os
import pickle



def load(emotion_label, specified_id = None):
    """ Loads a list with all the records corresponding to the given emotion label (label used at training time).
        A records is a list of tuples with two coords (one tuple per image): the movement parameter between 0 and 1, and the landmark as a numpy array.
        If a specified_id is given, only the record with this id is returned. """
    records = []
    folder = f'data/landmark_records/{emotion_label}'
    for f in os.listdir(folder):
        if f.startswith('record'):
            _,suffix = f.split('_')
            id,_ = suffix.split('.')
            id = int(id)
            if not specified_id or id == specified_id:
                with open(f'{folder}/{f}', 'rb') as pickle_in:
                    record = pickle.load(pickle_in)
                records.append(record)
                print(f)
    return records

records = load('angle_visage')
