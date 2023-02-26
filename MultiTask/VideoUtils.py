import os
import glob
import cv2
import time
import imutils


def write_video(video):

    frameNr = 0
    files = glob.glob('MultiTask/video_output/*')
    for f in files:
        os.remove(f)

    cv2.namedWindow('window')
    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        ret, frame = cap.read()

        frameNr += 1

        if ret:
            frame = imutils.resize(frame, width=1000)
            cv2.imwrite(f'video_output/frame_{frameNr}.png', frame)
            cv2.imshow("window", frame)

            key = cv2.waitKey(1)
            if key > 0:  # exit by pressing any key
                # destroy windows
                cv2.destroyAllWindows()

                for i in range(1, 5):
                    cv2.waitKey(1)
                return
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            key = cv2.waitKey(1)
            if key > 0:  # exit by pressing any key
                # destroy windows
                cv2.destroyAllWindows()

                # for i in range (1,5):
                #     cv2.waitKey(1)
                # return
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1e-20)

    cap.release()
    # cv2.destroyAllWindows()


def play_video(video):

    cv2.namedWindow("window")
    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            frame = imutils.resize(frame, width=1500, height=2500)
            cv2.imshow("window", frame)

            key = cv2.waitKey(20)
            if key > 0:  # exit by pressing any key
                # destroy windows
                cv2.destroyAllWindows()

                for i in range(1, 5):
                    cv2.waitKey(1)
                return
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # time.sleep(0.05)

    cap.release()
