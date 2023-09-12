from uuid import uuid4

import cv2
from pathlib import Path

import torch

from train_model import MobileNetV3RPS
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.transforms.v2 import AutoAugment, AutoAugmentPolicy, Compose, \
    RandomHorizontalFlip, RandomVerticalFlip

import random

import itertools
import numpy as np

def main():
    recognition()


def recognition():
    storage = {
        0: 'rock',
        1: 'paper',
        2: 'scissors'
    }
    storage_invert = {
        'rock': 0,
        'paper': 1,
        'scissors': 2
    }

    model_path = Path.home().joinpath('PycharmProjects', 'StudyWeek2023', 'experiments', '9895b659b3894132947cc1842ac79ae5', 'checkpoints', 'epoch=9-val_acc=0.83.ckpt')
    model = MobileNetV3RPS.load_from_checkpoint(model_path)
    m_net_transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    augmentation = Compose([
                AutoAugment(AutoAugmentPolicy.CIFAR10),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ])
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
        frame = cv2.rectangle(frame, capture_rec[0], capture_rec[1], (0, 0, 255))
        frame = cv2.putText(frame, "l - login answer", (capture_rec[0][0] + 30, capture_rec[0][1] - 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 64, 64), 3)
        frame = cv2.putText(frame, "q - Quit", (capture_rec[0][0] + 30, capture_rec[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 64, 64), 3)

        if key == ord('l'):
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imshow('test', crop)
            crop = torch.from_numpy(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ).permute(2, 0, 1)

            crop = augmentation(crop)
            input_tensor = m_net_transform(crop).unsqueeze(0)

            prediction = model(input_tensor)
            print(prediction)
            predicted_class_index = prediction.argmax(dim=1).item()
            guess = storage[predicted_class_index]
            frame = cv2.putText(frame, f'guess: {guess}', (capture_rec[0][0] + 30, capture_rec[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 64, 64), 3)
            cv2.imshow(window, frame)
            cv2.waitKey(3000)
            guess = storage_invert[guess]

            probabilities = np.random.multinomial(1, [1/3.378378378378378, 1/2.824858757062147, 1/2.857142857142857]*3)

            if probabilities[0] == 1:
                algorithm_guess = 1
            if probabilities[1] == 1:
                algorithm_guess = 2
            if probabilities[2] == 1:
               algorithm_guess = 0
            frame = cv2.putText(frame, f'I will choose {storage[algorithm_guess]}', (capture_rec[0][0] + 30, capture_rec[0][1] + 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 64, 64), 3)

            cv2.imshow(window, frame)
            cv2.waitKey(5000)

        else:
            cv2.imshow(window, frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
