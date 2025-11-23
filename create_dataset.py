import os
import pickle
import mediapipe as mp
import cv2

DATA_DIR = 'data'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

data = []
labels = []

for folder in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, folder)):
        continue
    if not folder.endswith("-samples"):
        continue

    label_name = folder.split("-")[0]
    class_path = os.path.join(DATA_DIR, folder)

    print("Processing:", folder)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            continue

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            z_ = [lm.z for lm in hand_landmarks.landmark]

            data_aux = []
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            min_z, max_z = min(z_), max(z_)

            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min_x) / (max_x - min_x) if max_x != min_x else 0.0
                norm_y = (lm.y - min_y) / (max_y - min_y) if max_y != min_y else 0.0
                norm_z = (lm.z - min_z) / (max_z - min_z) if max_z != min_z else 0.0
                data_aux.extend([norm_x, norm_y, norm_z])

            data.append(data_aux)
            labels.append(label_name)

print("Total samples:", len(data))

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Saved to data.pickle")
