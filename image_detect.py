import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time, os
import urllib.request

# url='sample_img/sample_person.jpg'


# %config InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 0.75
}

tfnet = TFNet(options)

img = cv2.imread(os.getcwd() + '\\sample_img\\sample_eagle.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
result = tfnet.return_predict(img)
# print(result)

for idx, res in enumerate(result):
    tl = (result[idx]['topleft']['x'], result[idx]['topleft']['y'])
    br = (result[idx]['bottomright']['x'], result[idx]['bottomright']['y'])
    label = result[idx]['label']


    # add the box and label and display it
    img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

plt.imshow(img)
plt.show()