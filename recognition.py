from uuid import uuid4

import cv2
from pathlib import Path

import torch

from train_model import MobileNetV3RPS
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.transforms.v2 import AutoAugment, AutoAugmentPolicy, Compose, \
    RandomHorizontalFlip, RandomVerticalFlip

import random
from random import randrange

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

    model_path = Path.home().joinpath('PycharmProjects', 'StudyWeek2023', 'experiments', 'f93cadc0cab94fd6b5982edd40abe34d', 'checkpoints', 'epoch=79-val_acc=0.93.ckpt')
    model = MobileNetV3RPS.load_from_checkpoint(model_path)
    model.eval()
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
    window = "AI Powered Rock, Paper, Scissors"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, 0, 0)
    # probabilities
    probabilities_rock = 3.378378378378378
    probabilities_paper = 2.824858757062147
    probabilities_scissors = 2.857142857142857
    is_most_probable = True
    algorithm_range = randrange(11)
    range_check = 0
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

        top_rect_start = (capture_rec[0][0], capture_rec[0][1] - 850)
        top_rect_end = (capture_rec[1][0], capture_rec[0][1])
        bottom_rect_start = (0, capture_rec[1][1])
        bottom_rect_end = (frame.shape[1], frame.shape[0])

        frame = cv2.rectangle(frame, top_rect_start, top_rect_end, (220, 218, 201), thickness=cv2.FILLED)
        frame = cv2.rectangle(frame, bottom_rect_start, bottom_rect_end, (124, 124, 124), thickness=cv2.FILLED)
        frame = cv2.rectangle(frame, bottom_rect_start, bottom_rect_end, (220, 218, 201), thickness=2)

        crop = frame.copy()[capture_rec[0][1]:capture_rec[1][1], capture_rec[0][0]:capture_rec[1][0]]
        frame = cv2.rectangle(frame, capture_rec[0], capture_rec[1], (0, 0, 0), thickness=2)
        frame = cv2.putText(frame, "Move here:", (capture_rec[0][0] + 10, capture_rec[0][1] + 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (64, 64, 64), 1)
        frame = cv2.putText(frame, "(L) Login answer", (capture_rec[0][0] + 10, capture_rec[0][1] - 85), cv2.FONT_HERSHEY_DUPLEX, 0.75, (64, 64, 64), 1)
        frame = cv2.putText(frame, "AI Rock, Paper, Scissors", (capture_rec[0][0] + 260, capture_rec[0][1] - 85), cv2.FONT_HERSHEY_DUPLEX, 0.75, (64, 64, 64), 1)
        frame = cv2.putText(frame, "(Q) Quit", (capture_rec[0][0] + 10, capture_rec[0][1] - 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (64, 64, 64), 1)
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        crop = torch.from_numpy(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1)

        crop = augmentation(crop)
        input_tensor = m_net_transform(crop).unsqueeze(0)

        prediction = model(input_tensor)
        print(prediction)
        predicted_class_index = prediction.argmax(dim=1).item()
        guess = storage[predicted_class_index]
        frame = cv2.putText(frame, f'Your guess: {guess}', (capture_rec[0][0] + 10, capture_rec[0][1] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (64, 64, 64), 1)
        cv2.imshow(window, frame)

        if key == ord('l'):
            guess = storage_invert[guess]
            probabilities_change = 0.2

            if guess == 0:
                probabilities_rock = 1/((1 / probabilities_rock) + probabilities_change)
                probabilities_paper = 1/((1 / probabilities_paper) - (probabilities_change/2))
                probabilities_scissors = 1/((1/probabilities_scissors) - (probabilities_change/2))
            if guess == 1:
                probabilities_paper = 1 / ((1 / probabilities_paper) + probabilities_change)
                probabilities_rock = 1 / ((1 / probabilities_rock) - (probabilities_change / 2))
                probabilities_scissors = 1 / ((1 / probabilities_scissors) - (probabilities_change / 2))
            if guess == 2:
                probabilities_scissors = 1 / ((1 / probabilities_scissors) + probabilities_change)
                probabilities_paper = 1 / ((1 / probabilities_paper) - (probabilities_change / 2))
                probabilities_rock = 1 / ((1 / probabilities_rock) - (probabilities_change / 2))

            probabilities = np.random.multinomial(1, [1/probabilities_rock, (1/probabilities_paper), (1/probabilities_scissors)])
            range_check += 1
            if range_check == algorithm_range:
                if is_most_probable:
                    is_most_probable = False
                if not is_most_probable:
                    is_most_probable = True
            algorithm_range = randrange(11)
            range_check = 0

            probabilities = np.random.multinomial(1, [1/3.378378378378378, (1/2.824858757062147), (1/2.857142857142857)])
            if is_most_probable:
                algorithm_guess = None
                if probabilities[0] == 1:
                    algorithm_guess = 1
                if probabilities[1] == 1:
                    algorithm_guess = 2
                if probabilities[2] == 1:
                    algorithm_guess = 0
            if not is_most_probable:
                algorithm_guess = None
                if probabilities[0] == 1:
                    algorithm_guess = 2
                if probabilities[1] == 1:
                    algorithm_guess = 0
                if probabilities[2] == 1:
                    algorithm_guess = 1

            frame = cv2.putText(frame, f'Computer: I will choose {storage[algorithm_guess]}', (capture_rec[0][0] + 10, capture_rec[0][1] + 290), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)
            cv2.imshow(window, frame)
            cv2.waitKey(3000)

            if algorithm_guess == guess:
                frame = cv2.putText(frame, 'Result: Its a tie no one wins', (capture_rec[0][0] + 10, capture_rec[0][1] + 320), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 0), 1)
            elif (algorithm_guess == 0 and guess == 2) or (algorithm_guess == 1 and guess == 0) or (algorithm_guess == 2 and guess == 1):
                frame = cv2.putText(frame, 'Result: I won muhahaha', (capture_rec[0][0] + 10, capture_rec[0][1] + 320), cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 255), 1)
            elif (algorithm_guess == 2 and guess == 0) or (algorithm_guess == 0 and guess == 1) or (algorithm_guess == 1 and guess == 2):
                frame = cv2.putText(frame, 'Result: You won', (capture_rec[0][0] + 10, capture_rec[0][1] + 320), cv2.FONT_HERSHEY_DUPLEX, 0.75, (26, 255, 10), 1)

            cv2.imshow(window, frame)
            cv2.waitKey(2000)

        else:
            cv2.imshow(window, frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
