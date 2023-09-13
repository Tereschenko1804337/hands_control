import math
import threading
from time import sleep

import cv2
import mediapipe as mp
import pyautogui
from screeninfo import get_monitors

# Инициализация
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

width = 1280
height = 720

x_RIGHT_FINGER_TIP, y_RIGHT_FINGER_TIP = 0, 0
x_center, y_center = 0, 0
smooth_x, smooth_y = 0, 0  # Для сглаживания
x_min, x_max, y_min, y_max = 0, 0, 0, 0  # Для сглаживания
alpha = 0.25  # Коэффициент сглаживания

lock = threading.Lock()
LeftClickDown = False
stop_program = False
stop_thread = False


def move_mouse():
    global smooth_x, smooth_y
    global x_center, y_center
    global x_min, x_max, y_min, y_max
    global stop_thread

    # Инициализация переменных для сглаживания
    init_smooth = False

    monitor = get_monitors()[1]  # Выбор второго монитора
    x, y, screen_width, screen_height = monitor.x, monitor.y, monitor.width, monitor.height
    while True:
        with lock:
            try:
                scaled_x = int((x_center - x_min) / (x_max - x_min) * screen_width)
                scaled_y = int((y_center - y_min) / (y_max - y_min) * screen_height)

                if x_center != 0 and y_center != 0:
                    # Инициализация переменных для сглаживания
                    if not init_smooth:
                        smooth_x, smooth_y = scaled_x, scaled_y
                        init_smooth = True

                    # Сглаживание
                    smooth_x = smooth_x * (1 - alpha) + scaled_x * alpha
                    smooth_y = smooth_y * (1 - alpha) + scaled_y * alpha

                    pyautogui.moveTo(smooth_x, smooth_y, duration=0)  # Используем smooth_x и smooth_y

                    if LeftClickDown:
                        pyautogui.click()
            except:
                pass

        sleep(0.015)

        if stop_thread:
            break


def start():
    global x_RIGHT_FINGER_TIP, y_RIGHT_FINGER_TIP
    global x_center, y_center
    global LeftClickDown
    global stop_program, stop_thread
    global x_min, x_max, y_min, y_max

    t2 = threading.Thread(target=move_mouse)
    t2.start()

    while not stop_program:
        success, image = cap.read()
        image = cv2.flip(image, 1)
        # Уменьшение размера изображения в два раза
        height, width = image.shape[:2]
        new_width = int(width / 2)
        new_height = int(height / 2)

        image = cv2.resize(image, (new_width, new_height))

        x_min = image.shape[1] * 0.4
        x_max = image.shape[1] * 0.6
        y_min = image.shape[0] * 0.4
        y_max = image.shape[0] * 0.6

        results = hands.process(image)

        x_LEFT_FINGER_TIP, y_LEFT_FINGER_TIP = 0, 0
        x_RIGHT_THUMB_TIP, y_RIGHT_THUMB_TIP = 0, 0

        if results.multi_hand_landmarks:
            for hand_num, handLms in enumerate(results.multi_hand_landmarks):  # working with each hand
                hand_info = results.multi_handedness[hand_num]
                hand_type = hand_info.classification[0].label
                if hand_type == "Right":
                    x_LEFT_FINGER_TIP, y_LEFT_FINGER_TIP = int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])
                elif hand_type == "Left":
                    x_RIGHT_FINGER_TIP, y_RIGHT_FINGER_TIP = int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])
                    x_RIGHT_THUMB_TIP, y_RIGHT_THUMB_TIP = int(handLms.landmark[mpHands.HandLandmark.THUMB_TIP].x * image.shape[1]), int(handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y * image.shape[0])

                x_center, y_center = 0, 0
                num_landmarks = 0

                for id, landmark in enumerate(handLms.landmark):
                    x_center += landmark.x * image.shape[1]
                    y_center += landmark.y * image.shape[0]
                    if id == 0:
                        z = landmark.z
                    num_landmarks += 1

                x_center /= num_landmarks
                y_center /= num_landmarks

                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

            distance = math.sqrt((x_RIGHT_FINGER_TIP - x_LEFT_FINGER_TIP) ** 2 + (y_RIGHT_FINGER_TIP - y_LEFT_FINGER_TIP) ** 2)
            if distance < 600:
                cv2.line(image, (x_LEFT_FINGER_TIP, y_LEFT_FINGER_TIP), (x_RIGHT_FINGER_TIP, y_RIGHT_FINGER_TIP), (0, 255, 0), 3)

            distance = math.sqrt((x_RIGHT_FINGER_TIP - x_RIGHT_THUMB_TIP) ** 2 + (y_RIGHT_FINGER_TIP - y_RIGHT_THUMB_TIP) ** 2)
            # print(f"[+] distance: {distance} | z: {z}")
            if distance < 20:
                LeftClickDown = True
                cv2.line(image, (x_RIGHT_FINGER_TIP, y_RIGHT_FINGER_TIP), (x_RIGHT_THUMB_TIP, y_RIGHT_THUMB_TIP), (0, 255, 255), 2)
            else:
                LeftClickDown = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_program = True
            stop_thread = True
            break

        resized_frame = cv2.resize(image, (width, height))
        cv2.imshow("Output", resized_frame)
        cv2.waitKey(1)
        sleep(0.01)

    t2.join()
    t2.stop()

start()

