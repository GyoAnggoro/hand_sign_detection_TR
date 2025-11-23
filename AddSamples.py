import cv2
import os

SAVE_DIR = "data/J-samples"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    cv2.imshow("Capture J-sample", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = f"J_{counter}.jpg"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        print("Saved:", filename)
        counter += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
