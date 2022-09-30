# Drishti
### Computer navigation using audio and video aid for Amputees and Parkinson’s patients
<h4>A virtual mouse pointer capable of moving cursor and performing cursor movement using facial landmarks and execute functions like right/left/double click using voice commands</h4>
<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#model-download">Model Download</a>
</p>

## Key Features
- **Control your cursor by moving your face**-
the predictor tracks the tip of your nose and moves the cursor in the direction the nose moves.
<p><h4>The User has the option to use either blinks or voice commands to execute functions like right/left/double clicks </h4></p>

- **Perform click functions by blinking**-
	- left eye blink– *left click*
	- Double left eye blink– *double left click*
	- right eye blink– *right click*

- **Alternatively, Perform click functions by the use of Voice Commands**-
	- Speak "Left Click"– *left click*
	- Speak "Double Click"– *double left click*
	- Speak "Right Click"– *right click*
  
 ## How To Use

To clone and run this application, you'll need [Python 3](https://www.python.org/) installed on your computer.
From your command line:

```bash
# Clone this repository
$ git clone https://github.com/ShauryaMagar/Drishti/

# Go into the repository
$ cd FaceMouse

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ python3 face_mouse.py 
```

## Credits

This software mainly uses the following open source packages (among others):

- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)
- [NumPy](https://numpy.org/)
- [PyAutoGUI](https://pypi.org/project/PyAutoGUI/)
- [Mouse](https://pypi.org/project/mouse/)
- [Vosk](https://alphacephei.com/vosk/)

## Model Download
Download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1jKVFurodcD15KKJ46iGMD1PfDmK0Ic13?usp=sharing). Place the folder in the root directory.
