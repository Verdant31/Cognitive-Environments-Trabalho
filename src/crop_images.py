import numpy as np
import os
import cv2

net = cv2.dnn.readNetFromCaffe(
    "../rois_model/deploy.prototxt",
    "../rois_model/res10_300x300_ssd_iter_140000.caffemodel"
)

fake_photos = os.listdir("../dataset/fake")
real_photos = os.listdir("../dataset/real")

saved = 0

for photo in real_photos:
    img = cv2.imread("../dataset/real/" + photo)
    h, w, _ = img.shape

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    if not os.path.exists("dataset/real"):
        os.makedirs("dataset/real")

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = img[startY:endY, startX:endX]

        cv2.imwrite("./dataset/real/img-" + str(saved) + ".png", face)
        saved += 1

saved = 0

for photo in fake_photos:
    img = cv2.imread("../dataset/fake/" + photo)
    h, w, _ = img.shape

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    if not os.path.exists("dataset/fake"):
        os.makedirs("dataset/fake")
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = img[startY:endY, startX:endX]

        if len(face) > 0:
            cv2.imwrite("./dataset/fake/img-" + str(saved) + ".png", face)
            saved += 1
