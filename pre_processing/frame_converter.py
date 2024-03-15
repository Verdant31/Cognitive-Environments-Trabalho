import cv2
import os

import numpy as np

videos = os.listdir("./videos")

net = cv2.dnn.readNetFromCaffe(
    "../rois_model/deploy.prototxt",
    "../rois_model/res10_300x300_ssd_iter_140000.caffemodel"
)


def video_to_frames(video_name, output_dir):
    cap = cv2.VideoCapture("./videos/" + video_name)
    frame_rate = 6
    frame_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        path = "/fake" if "Fake" in video_name else "/real"
        h, w, _ = frame.shape

        if frame_count % int(cap.get(5) / frame_rate) == 0:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                if len(face) > 0:
                    file_name = "frame_" + str(frame_count) + ".jpg"
                    cv2.imwrite(os.path.join(output_dir + path, file_name), face)

    cap.release()
    cv2.destroyAllWindows()


for video in videos:
    video_to_frames(video, "../dataset")
