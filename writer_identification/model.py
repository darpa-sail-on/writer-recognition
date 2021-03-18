from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D

# Function to resize image to 56x56
def resize_image(image):
    import tensorflow as tf
    return tf.image.resize(image,[56,56])

def build_model(num_classes, resize=False):
  # Build a neural network in Keras
  # Function to resize image to 64x64
  row, col, ch = 113, 113, 1

  model = Sequential()
  model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))
 
  if resize:
     #Resise data within the neural network
     model.add(Lambda(resize_image))  #resize images to allow for easy computation

  # CNN model - Building the model suggested in paper

  model.add(Convolution2D(filters= 32, kernel_size =(5,5), strides= (2,2), padding='same', name='conv1')) #96
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool1'))

  model.add(Convolution2D(filters= 64, kernel_size =(3,3), strides= (1,1), padding='same', name='conv2'))  #256
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool2'))

  model.add(Convolution2D(filters= 128, kernel_size =(3,3), strides= (1,1), padding='same', name='conv3'))  #256
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool3'))


  model.add(Flatten())
  model.add(Dropout(0.5))

  model.add(Dense(512, name='dense1'))  #1024
  # model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(256, name='dense2'))  #1024
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(num_classes,name='output'))
  model.add(Activation('softmax'))  #softmax since output is within 50 classes

  return model
