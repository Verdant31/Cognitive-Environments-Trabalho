import cv2
import os

videos = os.listdir("./videos")


def video_to_frames(video_name, output_dir):
    cap = cv2.VideoCapture("./videos/" + video_name)
    frame_rate = 3
    frame_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        path = "/fake" if "Fake" in video_name else "/real"

        if frame_count % int(cap.get(5) / frame_rate) == 0:
            file_name = "frame_" + str(frame_count) + ".jpg"
            cv2.imwrite(os.path.join(output_dir + path, file_name), frame)

    cap.release()
    cv2.destroyAllWindows()


for video in videos:
    video_to_frames(video, "../dataset")
