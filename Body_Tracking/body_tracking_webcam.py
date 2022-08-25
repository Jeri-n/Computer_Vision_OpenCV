import cv2
import mediapipe as mp
import numpy as np

mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils



cap=cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        succes,image=cap.read()
        if not succes:
            print("camera not opened")
            continue
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(720,480))

        results=pose.process(image)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows() 