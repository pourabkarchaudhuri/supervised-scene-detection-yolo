import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as  np

# %config InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 0.75
}

tfnet = TFNet(options)


img = cv2.imread('sample_imag/sample_dog.jpg', cv2.IMREAD_COLOR)
result = tfnet.return_predict(img)

img.shape

tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
label = result[0]

img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()