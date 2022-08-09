# import mediapipe - library that helps with machine learning built-in modules
import mediapipe as mp
# import opencv - access our webcam, computer vision tool
import cv2
import os  # folder control
import numpy as np  # array library
import pandas as pd
import pickle  # import model
from sklearn.model_selection import train_test_split


def mainScreen(result):
    path = r'C:\Users\WSM\Desktop\Speaky\Source code\main screen.jpeg'
    image = cv2.imread(path)  # load menu - a image
    image = cv2.putText(image, "Last score: ", (190, 820),  # show updated score
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, str(format(result, '.3f')), (190, 870),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.namedWindow(version,
                    cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(version, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(version, image)
    # while in the main screen the user got 4 options - 3 detection modes with different models and an exit option. the whole main menu functionality is
    # maintained by a menu JPEG and a listen event for the user input.
    if cv2.waitKey(0) & 0xFF == ord('1'):
        screen = modeList[0]
        # load the trained model
        with open(r'C:\Users\WSM\Desktop\Speaky\Source code\Models\Model___EyeContact___0.15v.pkl', 'rb') as f:
            model = pickle.load(f)
        Detection(screen, model)

    if cv2.waitKey(0) & 0xFF == ord('2'):
        screen = modeList[1]
        with open(r'C:\Users\WSM\Desktop\Speaky\Source code\Models\Model___BodyLanguage___0.16v.pkl', 'rb') as f:  # load the trained model
            model = pickle.load(f)
        Detection(screen, model)

    if cv2.waitKey(0) & 0xFF == ord('3'):
        screen = modeList[2]
        with open(r'C:\Users\WSM\Desktop\Speaky\Source code\Models\Model___MOOD___0.17v.pkl', 'rb') as f:  # load the trained model
            model = pickle.load(f)
        Detection(screen, model)

    if cv2.waitKey(0) & 0xFF == ord('4'):
        cv2.destroyAllWindows()  # close cv window
        print("Goodbye!")


def Detection(screen, model):
    counter = 0  # maintained counter to compute the score
    score = 0.0000001
    mp_drawing = mp.solutions.drawing_utils  # transfer to cv
    mp_holistic = mp.solutions.holistic  # mediapipe holistic module
    cap = cv2.VideoCapture(0)  # capture from web cam (0 in most PCs)
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

            # # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

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

                # detect
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                # print(body_language_class, body_language_prob)

                # Grab ear coords
                # coords = tuple(np.multiply(
                #     np.array(
                #         (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                #          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640, 480]).astype(int))

                # cv2.rectangle(image,
                #               (coords[0], coords[1]+5),
                #               (coords[0]+len(body_language_class)
                #                * 20, coords[1]-30),
                #               (245, 117, 16), -1)
                # cv2.putText(image, body_language_class, coords,
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(image, (0, 0), (1000, 30), (255, 0, 0), -1)
                # Display Speaky
                cv2.putText(image, version, (2, 18),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, '|||', (120, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class, (143, 21),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (
                    410, 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                counter = counter + 1  # to count number of frames we worked on
                if (body_language_class == targetList[0] or body_language_class == targetList[2] or body_language_class == targetList[4]):
                    score = float(score) + \
                        float(np.argmax(body_language_prob)
                              )  # add to score for positive
                else:
                    score = float(score) - 0.5  # penaltize score for negative
            except:
                pass
            cv2.namedWindow(version,
                            cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(version, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.imshow(version, image)
            if cv2.waitKey(10) & 0xFF == ord('4'):  # exit
                break
    cap.release()
    result = 0
    if ((score/counter) > 0):
        result = (score/counter)
    mainScreen(result)  # return score to main screen


cap = cv2.VideoCapture(0)
version = 'Speaky 0.2v'  # current speaky version
targetList = ["EYE CONTACT POSITIVE", "EYE CONTACT NEGATIVE",
              "BODY POSITIVE", "BODY NEGATIVE", "MOOD POSITIVE", "MOOD NEGATIVE"]  # 6 classfications, each pair to 1 model
modeList = ["Speaky - Eye contact detection", "Speaky - Body language detection",  # 3 mode names
            "Speaky - Mood detection"]
result = 0
mainScreen(result)
