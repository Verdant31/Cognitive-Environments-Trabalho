import numpy as np
import os
import cv2
from clear_files_names import clear_files_names

clear_files_names()

net = cv2.dnn.readNetFromCaffe(
    "../rois_model/deploy.prototxt",
    "../rois_model/res10_300x300_ssd_iter_140000.caffemodel"
)

fake_photos = os.listdir("./photos/fake")
real_photos = os.listdir("./photos/real")

photos_types = ["real", "fake"]

for photo_type in photos_types:
    saved = 0

    photos = real_photos if photo_type == "real" else fake_photos

    for photo in photos:
        img = cv2.imread("./photos/" + photo_type + "/" + photo)
        h, w, _ = img.shape

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        if not os.path.exists("../dataset/" + photo_type):
            os.makedirs("../dataset/" + photo_type)

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = img[startY:endY, startX:endX]

            cv2.imwrite("../dataset/" + photo_type + "/img-" + str(saved) + ".png", face)
            saved += 1

