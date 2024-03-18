from aws_client import client

mocked_users_from_database = [
    {"username": "default", "name": "4DTSR", "password": "default", "img": "./db/default_user.png"},
]


def authenticate(username, password):
    user = [user for user in mocked_users_from_database
            if user["username"] == username and user["password"] == password]

    return user
