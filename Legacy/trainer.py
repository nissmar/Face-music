import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import cv2


class Trainer:
    orientation = 0
    left_eye = None
    right_eye = None
    is_processing_left_eye = True

    def set_orientation(self, x):
        self.orientation = int(x)
        print(self.orientation)

    def validate(self, event):
        plt.close()
        try:
            with open('data/metadata.pickle', 'rb') as pickle_in:
                orientation_of_pictures = pickle.load(pickle_in)
        except EOFError:
            orientation_of_pictures = {}
        image_label = len(orientation_of_pictures)
        orientation_of_pictures[image_label] = self.orientation
        eye = self.left_eye if self.is_processing_left_eye else self.right_eye
        cv2.imwrite(f'data/images/{image_label}.jpeg', eye)
        with open('data/metadata.pickle', 'wb') as pickle_out:
            pickle.dump(orientation_of_pictures, pickle_out)


    def process_eye(self, eye):
        gig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        plt.imshow(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
        ax_validate = plt.axes([0.7, 0.05, 0.1, 0.075])
        validate_b = Button(ax_validate, 'Validate')
        validate_b.on_clicked(self.validate)
        ax_slider = plt.axes([0.3, 0.05, 0.2, 0.075])
        orientation_slider = Slider(ax_slider, 'Orientation', 0, 360)
        orientation_slider.on_changed(self.set_orientation)
        plt.show()

    def train(self, left_eye, right_eye):
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.is_processing_left_eye = True
        self.process_eye(left_eye)
        self.is_processing_left_eye = False
        self.process_eye(right_eye)
