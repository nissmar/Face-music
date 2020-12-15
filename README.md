# Face-music



## Face synthesizing 101

To use the face synthesizer, run `main.py`.

Rotate your face up/down and left/right to move the green dot on the screen. Each instrument has 4 distincts notes.

To switch instruments, tilt your head to the left/right. 

To start recording a loop, open your mouth. Closing it ends the recording. 

Want to toss your last recording? Simply raise your eyebrows.

## Calibration

We tailored the parameters of the program to our head, but each face is different. If you encounter some problems with the mouth/eyebrows movements, press `c`. Open your mouth a few times and press `c` again. Raise your eyebrows a few times and press `c` to finish the calibration. 

## Custom sound creation

All the sounds were generated with `sound_generator.py`. Feel free to create your own instruments and add them to the Face synthesizer!

## Head pose dataset.
To train our SVR, we used this dataset http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html?fbclid=IwAR0DnDN5iouR5WZ7GZ5U_AQhaogKmuKJrrY15k6lIEMQkocDwVzjW0v4k-g
