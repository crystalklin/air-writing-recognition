# Air Writing Recognition

Our project, Air Gesture Recognition, aims to use a combination of computer vision and handwriting recognition to create model that recognizes gestures written in air as text. Model users would be able to write “air words” facing a web-camera either real time or in-advance and have those gestures translated into character digits or words. 

To us, air gesture recognition is an interesting topic to create a solution for because it lends itself to many different future uses. Most intuitively, air gesture analysis allows for users with specific needs alternative forms of communication. This idea could also be extrapolated into using air gestures as a more universal input interface for technologies that are unsuited for the traditional keyboard and mouse (such as AR and VR systems).


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Environment

Python Version 3.7.x<br>
Packages:
* OpenCV 3.4.2
* imutils
* Tensorflow
* Keras
* numpy
* scikit-learn
* matplotlib

```
pip install opencv-python==3.4.2.16
pip install imutils
pip install tensorflow
pip install keras
pip install numpy
pip install scikit-learn
pip install matplotlib
```

### Project Code
You'll need to download the project repsository from GitHub:
```
https://github.com/heycrystal/air-writing-recognition
```

### Dataset

Download EMNIST Dataset at this address:

```
http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
```
The extract the following put them in a folder named 'data'
* emnist-byclass-test-images-idx3-ubyte.gz
* emnist-byclass-test-labels-idx1-ubyte.gz
* emnist-byclass-train-images-idx3-ubyte.gz
* emnist-byclass-train-labels-idx1-ubyte.gz


## Ready, set, run!
Our project can be run by training an new CNN entirely, or just with a presaved model. If you would like to just run the program without training a new model, feel free to skip to step 2.

###1. Training the classifier
To train the network you're going to need to run the file cnn_model_test.py. The program will create files cnn_model.json, cnn_model_weights.h5, and cnn_model_history.pckl. Using the "-v" or "--verbose" flag will show status print statements. Training takes us just under 30 minutes an epoch (total \~5 hours).

Note: The new model files will overwrite the old ones of the same name, so save the old ones under a different name if you'd like to keep them.
```
python cnn_model.py
```

###2. Plotting the classification history
To test and plot the saved model you will use the cnn_model_test.py file.
To plot the saved model accuracies and loss percentages during training, run:
```
python cnn_model_test.py -p
```
To test the saved model on the testing data from EMNIST, run:
```
python cnn_model_test.py -t
```
Once again, you may use the "-v" or "--verbose" flag to show status print statements.

###3. Running the program
To run the program you will need to run the air_writing_recognition.py program.
```
python air_writing_recognition.py
```
After running the program, two windows will pop up-- a webcam live feed and a black and white bitmask feed. (These two windows may be ontop of each other.) You will need a lime green object to track. <br>

To set the whiteboard depth, hit the "s" key. To write, simply start your letter and make lift your pen from the whiteboard between each stroke (increase the distance from the camera). To save the character, hit the "d" key. You may save multiple images in one session.

Note: During keypresses, you must be focused on the life feed window, _not_ the terminal window.

## Built With

* [OpenCV](http://www.opencv.com) - The computer vision library used

## Authors

* **Michelle Jin** - [_mjin8_](https://github.com/mjin8/)
* **Crystal Lin** - [_heycrystal_](https://github.com/heycrystal/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


