from aws_client import client


def verifyImage(buffer):
    response = client.detect_faces(
        Image={'Bytes': buffer},
        Attributes=["ALL"]
    )
    search_image_deformities(response)


def search_image_deformities(response):
    if len(response["FaceDetails"]) > 1:
        print("Imagem com 2 faces.")
        return None
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
