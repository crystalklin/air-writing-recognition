import keras
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json

from mnist import MNIST
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load trained model
print("Loading cnn model from disk.............", end="")
json_file = open('cnn_model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("cnn_model_weights.h5")
print("...finished.")

# Setting variables
print("Setting constants.......................", end="")
batch_size = 128
num_classes = 26 + 26 + 10  # 26 lower, 26 upper, 10 digits
epochs = 10
img_rows, img_cols = 28, 28 # image dimensions (28 x 28 pixels)
print("...finished.")

# Load EMNIST-byclass data
print("Loading EMNIST byclass-training data....", end="")
emnist = MNIST(path='data', return_type='numpy')
emnist.select_emnist('byclass')
X, y =  emnist.load_training()
print("...finished.")

# Split test data into training data and evaluation data
print("Split data: training and test(eval).....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=111)
print("...finished.")
print("\tTraining Shape:", y_train.shape)
print("\tTesting Shape:", y_test.shape)

# Restructuring data (batch, steps, channels)
print("Restructuring data...", end="")
input_shape = None
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    print("..Channels first...", end="")
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    print("..Channels last...", end="")

# Change pixel information type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Remap [0,255] image values to [0,1]
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices
'''
categorical[np.arange(n), y] = 1
IndexError: index 255 is out of bounds for axis 1 with size 62
'''
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("...finished.")

# Compile model
print("Compiling model.........................", end="")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print("...finished.")

# Evaluate the model using Accuracy and Loss
print("Evaluating model........................", end="")
score = model.evaluate(X_test, y_test, verbose=0)
print("...finished.")
print('\tTest loss:', score[0])
print('\tTest accuracy:', score[1])










