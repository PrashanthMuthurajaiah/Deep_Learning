# Imports

import cv2 # Open CV
import numpy as np # Linear Algebra
import argparse # Argument Parser

# Construct arguement parsers

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required = True, help="path to prototxt file")
ap.add_argument("-m", "--model", required = True, help="path to caffe model weights")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter out weak detections")
args = vars(ap.parse_args())

# Load serialized models and their weights
print("[INFO] Loading the model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
print("[INFO] Model loaded.")


# Read the image and convert to image to image blob

img = cv2.imread(args['image'])
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Computing object detection
print("[INFO] Computing object detection")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter based on confidence
    if confidence > args['confidence']:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
		# probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
