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
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

    
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dense(activation = 'sigmoid', units=6))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])





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


classifier.fit_generator(
        training_set,
        steps_per_epoch=500,
        epochs=30,
        validation_steps = 50,
        validation_data=test_set)



classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("classifier.h5")
print("Saved model to disk")

model_json = classifier.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)

classifier.save_weights("model_weights.h5")


#later
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")


