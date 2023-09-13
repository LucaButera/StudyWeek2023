import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from train_model import MobileNetV3RPS, RPSDatamodule


class Denormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)


def main():
    denorm = Denormalize()
    storage = {
        0: 'r',
        1: 'p',
        2: 's'
    }
    dm = RPSDatamodule(batch_size=1, augmentation=False)
    dm.setup()
    model_path = ''
    model = MobileNetV3RPS.load_from_checkpoint(model_path)
    model.eval()
    correct, overall = 0, 0
    for img, gt in dm.test_dataloader():
        pred = model(img).argmax(dim=1)[0].item()
        if pred == gt.item():
            correct += 1
        overall += 1
        pic = cv2.cvtColor((denorm(img)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(pic, f"gt: {storage[gt[0].item()]}, pr: {storage[pred]}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('test', pic)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    print('Accuracy: ' + str(correct/overall))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()