# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:06:09 2020

@author: truls
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape = (64,64,3), activation = 'relu'))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(32, (3, 3), activation='relu'))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))


classifier.add(Flatten())
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dense(activation = 'softmax', units=6))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =['accuracy'])





train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )


test_datagen = ImageDataGenerator (rescale=1./255)


training_set = train_datagen.flow_from_directory(
        'dataset/train',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/valid',
        target_size =(64,64),
        batch_size = 32,
        class_mode = 'categorical')

print(training_set.class_indices)

classifier.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=30,
        validation_steps = 30,
        validation_data=test_set)



classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("classifier.h5")
print("Saved model to disk")


test_image = image.load_img('d6.jpg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image) 

print(result)