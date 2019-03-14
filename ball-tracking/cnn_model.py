import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K

from mnist import MNIST     # to load emnist data
from sklearn.model_selection import train_test_split # splitting data
import pickle               # saving training history
import matplotlib.pyplot as plt # plotting the history
import argparse             # handle argument flags

# Set commandline flags
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity", 
    action="store_true")
parser.add_argument("-p", "--plot", help="plot training accuracy and loss",
    action="store_true")
parser.add_argument("-t", "--test", help="run model on test data",
    action="store_true")

args = parser.parse_args()
if args.verbose:
    print("Turned on: verbosity")
if args.plot:
    print("Turned on: plot")
if args.test:
    print("Turned on: test")

# Setting variables
if args.verbose:
    print("Setting constants.......................", end="")
batch_size = 128
num_classes = 26 + 26 + 10  # 26 lower, 26 upper, 10 digits
epochs = 10
img_rows, img_cols = 28, 28 # image dimensions (28 x 28 pixels)
if args.verbose:
    print("...finished.")

# Load EMNIST-byclass data
if args.verbose:
    print("Loading EMNIST byclass-training data....", end="")
emnist = MNIST(path='data', return_type='numpy')
emnist.select_emnist('byclass')
X, y =  emnist.load_training()
if args.verbose:
    print("...finished.")

# Split test data into training data and evaluation data
if args.verbose:
    print("Split data: training and test(eval).....", end="")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=111)
if args.verbose:
    print("...finished.")
#print("\tTraining Shape:", y_train.shape)
#print("\tTesting Shape:", y_test.shape)

# Restructuring data (batch, steps, channels)
if args.verbose:
    print("Restructuring data...", end="")
input_shape = None
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    if args.verbose:
        print("...Channels first...", end="")
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    if args.verbose:
        print("...Channels last...", end="")

# Change pixel information type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Remap [0,255] image values to [0,1]
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
if args.verbose:
    print("...finished.")

# Create network architecture
if args.verbose:
    print("Building network architecture...........", end="")
model = Sequential()

# add input convolutional layer
# 32 3x3 filters, stride 1, ReLU
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# add convolutional layer
# 64 3x3 filters, stride 1, ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))

# add maxpooling, dropout, flatten layers
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

# Add feedforward hidden layer
# 128 nodes, ReLU
model.add(Dense(128, activation='relu'))

# Add dropout, softmax layer
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
if args.verbose:
    print("...finished.")

# Compile model
if args.verbose:
    print("Compiling model.........................", end="")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
if args.verbose:
    print("...finished.")

# Train model
if args.verbose:
    print("Training model...")
history = model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test))
if args.verbose:
    print("...finished.")
print(hist.history)

# Save trained model
if args.verbose:
    print("Saving model to disk....................", end="")

# Save model in JSON format
model_json = model.to_json()
with open("cnn_model.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save('cnn_model_weights.h5')

# Save training history
f = open('cnn_model_history.pckl', 'wb')
pickle.dump(history.history, f)
f.close()
if args.verbose:
    print("...finished.")

if args.plot:
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if args.test:
    # Evaluate the model using Accuracy and Loss
    if args.verbose:
        print("Evaluating model........................", end="")
    score = model.evaluate(X_test, y_test, verbose=0)
    if args.verbose:
        print("...finished.")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

