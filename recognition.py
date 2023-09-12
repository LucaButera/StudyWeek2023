import cv2
import torch
from pathlib import Path
from train_model import MobileNetV3RPS
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.transforms.v2 import AutoAugment, AutoAugmentPolicy, Compose, RandomHorizontalFlip, RandomVerticalFlip
import random


def main():
    r_p_s_algorithm()


def recognition(algorithm_guess):
    storage = {
        0: 'rock',
        1: 'paper',
        2: 'scissors'
    }

    model_path = Path.home().joinpath('PycharmProjects', 'StudyWeek2023', 'experiments', '5ec65032805440489a7888c997ff4492', 'checkpoints', 'epoch=93-val_acc=0.93.ckpt')
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

        text_position1 = (capture_rec[0][0] + 30, capture_rec[0][1] - 200)
        text_position2 = (capture_rec[0][0] + 30, capture_rec[0][1] - 50)
        text_position3 = (capture_rec[1][0] + 30, capture_rec[0][1] - 50)
        text_position4 = (capture_rec[1][0] + 30, capture_rec[1][1] + 30)

        for text_position in [text_position1, text_position2, text_position3, text_position4]:
            if text_position[1] < capture_rec[0][1]:
                text_position = (text_position[0], capture_rec[0][1] + 30)

        frame = cv2.rectangle(frame, capture_rec[0], capture_rec[1], (0, 0, 255))
        frame = cv2.putText(frame, "l - login answer", text_position1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        frame = cv2.putText(frame, "q - Quit", text_position2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

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

            guess_text_position = (capture_rec[0][0] + 30, capture_rec[0][1] - 50)

            if guess_text_position[1] < text_position2[1]:
                guess_text_position = (guess_text_position[0], text_position2[1] + 30)

            frame = cv2.putText(frame, f'guess: {guess}', guess_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow(window, frame)
            cv2.waitKey(2000)

            frame = cv2.putText(frame, f'I will choose {storage[algorithm_guess]}', guess_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            if algorithm_guess == guess:
                frame = cv2.putText(frame, 'Its a tie no one wins ', guess_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            elif (algorithm_guess == 0 and guess == 2) or \
                    (algorithm_guess == 1 and guess == 0) or \
                    (algorithm_guess == 2 and guess == 1):
                frame = cv2.putText(frame, 'I won muhahaha', guess_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            else:
                frame = cv2.putText(frame, 'You won ☹', guess_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            cv2.imshow(window, frame)
            cv2.waitKey(2000)

        else:
            cv2.imshow(window, frame)
    cap.release()
    cv2.destroyAllWindows()


def r_p_s_algorithm():
    choice = [0, 1, 2]
    recognition(random.choice(choice))


if __name__ == "__main__":
    main()
