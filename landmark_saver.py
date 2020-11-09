
from sound_training import play
import time
import cv2

def display_time(img, mu):
    x1,x2,y1,y2 = 10,50,10,20
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
    #img = cv2.rectangle(img, (x1+1,y1+1), (x1+mu*(x2-x1-2),y2-1), (255,255,255), 1)
    return img



class LandmarkSaver:
    capture_length = 10

    def begin_capture(self):
        self.capture_begin = time.time()

    def poke(self, img, landmark):
        t = time.time()
        mu = (t - self.capture_begin) / self.capture_length
        capture_has_ended =  mu > 1
        if capture_has_ended:
            return False, img
        return True, display_time(img, mu)
            
