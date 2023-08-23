import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

color_mouse_pointer = (255, 0, 255)

# Ajusta estas coordenadas según las dimensiones de tu pantalla de laptop
SCREEN_LAPTOP_X_INI = 120
SCREEN_LAPTOP_Y_INI = 130
SCREEN_LAPTOP_X_FIN = 150 + 1366  # Ancho de la pantalla de tu laptop
SCREEN_LAPTOP_Y_FIN = 160 + 768  # Alto de la pantalla de tu laptop

X_Y_INI = 100

def calculate_distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)

def detect_finger_down(hand_landmarks, width, height):
    # Resto del código de la función detect_finger_down...

# Resto del código...

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)
                xm = np.interp(x, (X_Y_INI, X_Y_INI + width), (SCREEN_LAPTOP_X_INI, SCREEN_LAPTOP_X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + height), (SCREEN_LAPTOP_Y_INI, SCREEN_LAPTOP_Y_FIN))
                pyautogui.moveTo(int(xm), int(ym))
                if detect_finger_down(hand_landmarks, width, height):
                    pyautogui.click()
                cv2.circle(frame, (x, y), 10, color_mouse_pointer, 3)
                cv2.circle(frame, (x, y), 5, color_mouse_pointer, -1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
