from uuid import uuid4

import cv2
from pathlib import Path


def main():
    root = Path.home().joinpath('StudyWeek2023', 'dataset')
    root.mkdir(parents=False, exist_ok=True)
    storage = {
        ord('r'): root / 'rock',
        ord('p'): root / 'paper',
        ord('s'): root / 'scissors',
    }
    for p in storage.values():
        p.mkdir(parents=False, exist_ok=True)
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
        capture_rec = ((0, c_y - 256), (512, c_y + 256))
        font_pos = (capture_rec[0][0] + 20, capture_rec[1][1] - 20)
        capture = key in [ord('s'), ord('p'), ord('r')]
        crop = frame.copy()[capture_rec[0][1]:capture_rec[1][1], capture_rec[0][0]:capture_rec[1][0]]

        frame = cv2.rectangle(frame, capture_rec[0], capture_rec[1], (0, 0, 255) if capture else (0, 255, 0), 3)
        frame = cv2.putText(frame, "r - Capture 'rock'", (capture_rec[0][0] + 30, capture_rec[0][1] - 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        frame = cv2.putText(frame, "p - Capture 'paper'", (capture_rec[0][0] + 30, capture_rec[0][1] - 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        frame = cv2.putText(frame, "s - Capture 'scissors'", (capture_rec[0][0] + 30, capture_rec[0][1] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        frame = cv2.putText(frame, "q - Quit", (capture_rec[0][0] + 30, capture_rec[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if capture:
            frame = cv2.putText(frame, chr(key).upper(), font_pos, cv2.FONT_HERSHEY_SIMPLEX, 22, (0, 255, 0), 7)
        cv2.imshow(window, frame)
        if capture:
            dest = storage[key]
            filename = f'{dest.name}_{uuid4().hex}.png'
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest/filename), crop)
            cv2.waitKey(750)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
