
# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from PIL import Image
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

 
# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     # load the image
     data = pyplot.imread(filename)
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     pyplot.show()
 
# define the test configuration

 
    
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("classifier.h5")
print("Loaded model from disk")

# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
test_image = image.load_img('d20.jpg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# 
# #predict the result
results = model.predict(test_image)
# print(result)

# load photograph

# visualize the results
draw_image_with_boxes('elephant.jpg', results[0]['rois'])