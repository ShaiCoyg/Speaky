# import mediapipe - library that helps with machine learning built-in modules
import mediapipe as mp
# import opencv - access our webcam, computer vision tool
import cv2
import csv  # write data to csv
import os  # folder control
import numpy as np  # array library

mp_drawing = mp.solutions.drawing_utils  # transfer to cv
mp_holistic = mp.solutions.holistic  # mediapipe holistic module
targetList = ["EYE CONTACT POSITIVE", "EYE CONTACT NEGATIVE",
              "BODY POSITIVE", "BODY NEGATIVE", "MOOD POSITIVE", "MOOD NEGATIVE"]

# first row of csv file(CLASS, x1, y1, z1, i, x2...)
# RUN ONE TIME ONLY FOR EACH CSV FILE
# comments ctrl+k+c, remove ctrl+k+u

landmarks = ['CLASS']
for val in range(1, 502):
    landmarks += ['x{}'.format(val), 'y{}'.format(val),
                  'z{}'.format(val), 'v{}'.format(val)]

with open(r'C:\Users\WSM\Desktop\Speaky\Source code\Coordinates\Coordinates___MOOD E___0.14v.csv', 'a') as f:
    csv_writer = csv.writer(
        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


cap = cv2.VideoCapture(0)  # capture from web cam (0 in most PCs)
# using holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():  # while web cam on
        ret, frame = cap.read()  # take the web cam output
        # change image to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)
        # Recolor image back to BGR for openCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            # Concate rows
            row = pose_row+face_row
            # Append class name
            row.insert(0, targetList[5])

            # Export to CSV
            with open(r'C:\Users\WSM\Desktop\Speaky\Source code\Coordinates\Coordinates___MOOD E___0.14v.csv', 'a') as csvfile:
                csv_writer = csv.writer(
                    csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
        except:
            pass
        cv2.imshow('Web Cam', image)  # popup window
        if cv2.waitKey(10) & 0xFF == ord('q'):  # to exit window and shut down camera
            break

cap.release()  # empty capture var
cv2.destroyAllWindows()  # close cv window
