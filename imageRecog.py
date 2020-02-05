# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:25:39 2020

@author: truls
"""
import cv2
from PIL import Image
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


#later
# load json and create model
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("classifier.h5")
print("Loaded model from disk")
# =============================================================================
# 
#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
test_image = image.load_img('d8.jpg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )

generator= train_datagen.flow_from_directory("dataset/train", batch_size=32)
label_map = (generator.class_indices)
print(label_map)
label_map = {value:key for key, value in label_map.items()}
result = model.predict_classes(test_image) 
prob = model.predict_proba(test_image)
print(prob)

#y_classes = y_prob.argmax(axis=-1)
print(label_map.get(result[0]))

# =============================================================================

#https://medium.com/@jinilcs/a-simple-keras-model-on-my-laptop-webcam-dda77521e6a0
# =============================================================================
#cv2.namedWindow("preview")
#video = cv2.VideoCapture(1)
#while True:
#        _, frame = video.read()
#
#        #Convert the captured frame into RGB
#        im = Image.fromarray(frame, 'RGB')
#
#        #Resizing into 128x128 because we trained the model with this image size.
#        im = im.resize((64,64))
#        img_array = np.array(im)
#
#        #Our keras model used a 4D tensor, (images x height x width x channel)
#        #So changing dimension 128x128x3 into 1x128x128x3 
#        #img_array = np.expand_dims(img_array, axis=0)
#
#        #Calling the predict method on model to predict 'me' on the image
#        prediction = int(model.predict(img_array)[0][0])
#
#        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
#        if prediction == 0:
#                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#        cv2.imshow("Capturing", frame)
#        key=cv2.waitKey(1)
#        if key == ord('q'):
#                break
#video.release()
#cv2.destroyAllWindows()
 # =============================================================================
