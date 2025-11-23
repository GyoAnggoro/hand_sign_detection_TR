import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
label_list = model_dict.get('labels', None)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
recent_preds = deque(maxlen=10)

output_text = ""
last_hand_time = None
hand_start_time = None
last_letter_time = None
underscore_added = False

DETECTION_DELAY = 1.5
LETTER_REPEAT_DELAY = 1.0
UNDERSCORE_DELAY = 2.0
CLEAR_DELAY = 4.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        last_hand_time = current_time
        underscore_added = False

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            z_ = [lm.z for lm in hand_landmarks.landmark]

            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            min_z, max_z = min(z_), max(z_)

            data_aux = []

            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min_x) / (max_x - min_x) if max_x != min_x else 0.0
                norm_y = (lm.y - min_y) / (max_y - min_y) if max_y != min_y else 0.0
                norm_z = (lm.z - min_z) / (max_z - min_z) if max_z != min_z else 0.0
                data_aux.extend([norm_x, norm_y, norm_z])

            data_aux = np.asarray(data_aux).reshape(1, -1)

            if hand_start_time is None:
                hand_start_time = current_time

            if current_time - hand_start_time >= DETECTION_DELAY:
                pred = model.predict(data_aux)
                predicted_label = pred[0]

                recent_preds.append(predicted_label)

                stable_label = max(set(recent_preds), key=recent_preds.count)

                if last_letter_time is None or current_time - last_letter_time >= LETTER_REPEAT_DELAY:
                    output_text += stable_label
                    last_letter_time = current_time
                    recent_preds.clear()
                    hand_start_time = current_time

    else:
        hand_start_time = None

        if last_hand_time is not None:
            elapsed = current_time - last_hand_time

            if elapsed >= UNDERSCORE_DELAY and elapsed < CLEAR_DELAY and not underscore_added:
                output_text += "_"
                underscore_added = True

            elif elapsed >= CLEAR_DELAY:
                output_text = ""
                last_hand_time = None
                underscore_added = False
                last_letter_time = None

    cv2.putText(frame, output_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ASL Continuous Prediction", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
