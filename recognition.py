from uuid import uuid4

import cv2
from pathlib import Path

import torch

from train_model import MobileNetV3RPS

def main():

    storage = {
        0: 'rock',
        1: 'paper',
        2: 'scissors'
    }

    # model_path = Path.home().joinpath('PycharmProjects', 'StudyWeek2023', 'experiments', 'eea5d83be49d46f2953b2739422a2b61', 'checkpoints')
    # model = torch.load(model_path)
    model = MobileNetV3RPS

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    window = "Acquisition"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, 0, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        frame = cv2.flip(frame, 1)  # Flip around y
        c_y, c_x = frame.shape[0] // 2, frame.shape[1] // 2
        capture_rec_size = 256
        capture_rec = ((0, c_y - capture_rec_size // 2), (capture_rec_size, c_y + capture_rec_size // 2))

        crop = frame.copy()[capture_rec[0][1]:capture_rec[1][1], capture_rec[0][0]:capture_rec[1][0]]

        frame = cv2.rectangle(frame, capture_rec[0], capture_rec[1], (0, 0, 255))
        frame = cv2.putText(frame, "l - login answer", (capture_rec[0][0] + 30, capture_rec[0][1] - 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        frame = cv2.putText(frame, "q - Quit", (capture_rec[0][0] + 30, capture_rec[0][1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        if key == ord('l'):
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            crop = crop.transpose((2, 0, 1))
            crop = crop / 255.0
            input_tensor = torch.from_numpy(crop).unsqueeze(0).float()
            prediction = model(input_tensor)
            predicted_class_index = prediction.argmax(prediction).item()
            guess = storage[predicted_class_index]
            frame = cv2.putText(frame, f'guess: {guess}', (capture_rec[0][0] + 30, capture_rec[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.waitKey(750)
        cv2.imshow(window, frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
