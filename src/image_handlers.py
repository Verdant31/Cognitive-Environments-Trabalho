import cv2
import numpy as np
from keras.src.utils import img_to_array
from tensorflow.keras.models import load_model
from aws_client import client
import pickle


def check_similarity(user, buffer):
    with open(user["img"], "rb") as file:
        img_file = file.read()
        bytes_file_target = bytearray(img_file)

    response = client.compare_faces(
        SourceImage={'Bytes': buffer},
        TargetImage={'Bytes': bytes_file_target},
    )

    if len(response["UnmatchedFaces"]) > 0:
        raise Exception("Your face was not related to any face of your database.")


def verify_image_liveness(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    net = cv2.dnn.readNetFromCaffe(
        "./rois_model/deploy.prototxt",
        "./rois_model/res10_300x300_ssd_iter_140000.caffemodel"
    )

    print("[INFO] loading liveness detector...")
    model = load_model("./src/liveness_model/model")
    le = pickle.loads(open("./src/liveness_model/le.pickle", "rb").read())

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")

        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, end_x)
        end_y = min(h, end_y)

        face = image[start_y:end_y, start_x:end_x]
        face = cv2.resize(face, (32, 32))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)[0]
        j = np.argmax(preds)
        label = le.classes_[j]

        label = "{}: {:.4f}".format(label, preds[j])
        cv2.putText(image, label, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                      (0, 0, 255), 2)

        return le.classes_[j], preds[j]


def search_image_deformities(response):
    if len(response["FaceDetails"]) > 1:
        raise Exception("More than two faces were detected in the uploaded photo.")

    face = response["FaceDetails"][0]
    eyes_open = face["EyesOpen"]
    sunglasses = face["Sunglasses"]
    occlusion = face["FaceOccluded"]

    if not eyes_open["Value"]:
        raise Exception("Please keep your eyes open while taking the picture")
    if sunglasses["Value"]:
        raise Exception("Please remove any type of glasses you may be wearing")
    if occlusion["Value"]:
        raise Exception("Take the photo in a well-lit environment")
