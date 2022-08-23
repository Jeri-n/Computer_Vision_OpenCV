import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        success,image = cap.read()

        if not success:
            print("Camera not open")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = hand_landmarks,
                    connections = mp_hands.HAND_CONNECTIONS
                )
                print(results.multi_handedness)
        cv2.imshow('tracking',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

