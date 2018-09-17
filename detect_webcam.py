import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time, os



# %config InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 0.75
}

tfnet = TFNet(options)

colors = [tuple(255*np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
# Put 0 for primary set camera (attached webcam/USB device)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

fps = 0
frame_num = 0

while True:
    start_time = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            print(result)
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence*100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

            frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)

        end_time = time.time()
        fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
        start_time = end_time

        frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
        cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        # print('FPS {:.1f}'.format(1/(time.time() - stime())))
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


capture.release()
cv2.destroyAllWindows()