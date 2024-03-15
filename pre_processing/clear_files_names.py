import os
from unidecode import unidecode

fake_photos = os.listdir("./photos/fake")
real_photos = os.listdir("./photos/real")

photos_types = ["real", "fake"]


def clear_files_names():
    for photo_type in photos_types:
        photos = real_photos if photo_type == "real" else fake_photos

        for photo in photos:
            photo_path = "./photos/" + photo_type + "/"
            new_photo_name = unidecode(photo.replace(" ", "").replace("-", "_").replace("#", ""))

            os.rename(src=photo_path + photo, dst=photo_path + new_photo_name)
