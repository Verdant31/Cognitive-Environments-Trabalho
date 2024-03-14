from aws_client import client

mocked_users_from_database = [
    {"username": "admin", "password": "admin", "img": "./database/admin.jpg"},
    {"username": "default", "password": "default", "img": "./database/default_user.jpg"}
]


def authenticate(username, password):
    user = [user for user in mocked_users_from_database
            if user["username"] == username and user["password"] == password]

    # if(len(user) > 0):
    #   with open(user[0]["img"], "rb") as file:
    #     img_file = file.read()
    #     bytes_file_target = bytearray(img_file)

    #   response = client.compare_faces(
    #     SourceImage={'Bytes': buffer},
    #     TargetImage={'Bytes': bytes_file_target},
    #   )
    #   print(response["FaceMatches"][0]["Similarity"])
    #   print(response["UnmatchedFaces"][0]["Similarity"])

    return user
