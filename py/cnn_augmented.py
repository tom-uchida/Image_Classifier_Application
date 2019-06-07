import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

classes = ["building", "person", "tree"]
num_of_classes = len(classes)
image_size = 50



def main():
    # Read .npy file
    X_train, X_test, y_train, y_test = np.load("./image_data_aug.npy")
    X_train = X_train.astype("float") / 256.0
    X_test  = X_test.astype("float") / 256.0

    # Convert to one-hot-vector
    # [0,1,2] → [1,0,0] [0,1,0] [0,0,1]
    y_train = np_utils.to_categorical(y_train, num_of_classes) 
    y_test  = np_utils.to_categorical(y_test, num_of_classes) 

    # Train model
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)



def model_train(_X_train, _y_train):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape = _X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(  loss='categorical_crossentropy', 
                    optimizer=opt, 
                    metrics=['accuracy'])

    model.fit(_X_train, _y_train, batch_size=32, epochs=1)

    # Save model
    model.save('./cnn_model_aug.h5')

    return model


def model_eval(_model, _X_test, _y_test):
    scores = _model.evaluate(_X_test, _y_test, verbose=1)
    print("Test Loss: ", scores[0])
    print("Test Accuracy: ", scores[1])



if __name__ == "__main__":
    main()