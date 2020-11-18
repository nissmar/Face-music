import time
import cv2
import os
import pickle

def display_time(img, mu, dx=0):
    """mu in [0,1]"""
    x1,x2,y1,y2 = 30+dx,200+dx,30,70
    
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
    x1,y1,x2,y2 = x1+1,y1+1,int(x1+mu*(x2-x1-2)),y2-1
    cv2.rectangle(img,(x1,y1), (x2,y2),(255,255,255), -1)
    return img


""" The landmark saver is used by features.py to save records of the landmarks during a short period.
    When you first hit the space button, the camera will freeze and you will need to enter a label corresponding to the emotion you want to record.
    When you're ready, hit the space button again and do the movement you want to record, following the slider speed.
    
    /!\ The length of the record is given by the class parameter record_length. 
        The metric associated to the stored landmarks in the record is mu = (time since record started)/record_length"""

class LandmarkSaver:

    # length of record (in seconds)
    record_length = 3

    def begin_record(self):
        self.record_begin = time.time()
        self.data = []

    def end_record(self):
        #create folder if needed
        folder = f"data/landmark_records/{self.emotion_label}"
        if not os.path.exists(folder):
            os.mkdir(folder)
        existing_id = []
        for f in os.listdir(folder):
            if f.startswith('record'):
                _,suffix = f.split('_')
                id,_ = suffix.split('.')
                existing_id.append(int(id))
        if not existing_id:
            record_id = 0
        else:
            record_id = min(list(set(range(max(existing_id)+2)) - set(existing_id)))
        with open(f"{folder}/record_{record_id}.pickle", 'wb') as pickle_out:
            pickle.dump(self.data, pickle_out)
        


    def poke(self, img, landmark):
        t = time.time()
        mu = (t - self.record_begin) / self.record_length
        if landmark is not None:
            self.data.append((mu,landmark))
        record_has_ended =  mu > 1
        if record_has_ended:
            self.end_record()
            return False, img
        return True, display_time(img, mu)

    def set_emotion_label(self, label):
        self.emotion_label = label
            

def load_landmark(emotion_label, specified_id = None):
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