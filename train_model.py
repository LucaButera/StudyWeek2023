from itertools import chain
from pathlib import Path
from uuid import uuid4

import cv2
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from numpy.random import default_rng
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchmetrics import Accuracy
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms.v2 import AutoAugment, AutoAugmentPolicy, Compose, \
    RandomHorizontalFlip, RandomVerticalFlip


class RPSDataset(Dataset):

    cls2int = {
        'rock': 0,
        'paper': 1,
        'scissors': 2,
    }

    def __init__(
        self,
        samples: list[Path],
        augmentation: bool = True,
    ) -> None:
        super().__init__()
        self.samples = samples
        if augmentation:
            self.augmentation = Compose([
                AutoAugment(AutoAugmentPolicy.CIFAR10),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ])
        else:
            self.augmentation = None
        self.m_net_transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()

    @classmethod
    def splits_from_dir(cls, data_dir: str | Path = Path.home().joinpath('PycharmProjects', 'StudyWeek2023', 'dataset')) -> dict[str, list[Path]]:
        samples = {}
        for d in data_dir.iterdir():
            if d.is_dir():
                samples[d.name] = []
                for f in d.iterdir():
                    if f.is_file() and f.suffix == ".png":
                        samples[d.name].append(f.absolute())
        splits = {'train': [], 'val': [], 'test': []}
        rng = default_rng(42)
        for v in samples.values():
            rng.shuffle(v)
            s1, s2 = int(0.1 * len(v)), int(0.2 * len(v))
            splits['test'].extend(v[:s1])
            splits['val'].extend(v[s1:s1+s2])
            splits['train'].extend(v[s1+s2:])
        return splits

    def __len__(self):
        return len(self.samples)

    def load_sample(self, sample: Path) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(
            cv2.cvtColor(cv2.imread(str(sample)), cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1)
        clazz = torch.tensor(self.cls2int[sample.stem.split('_')[0]])
        if self.augmentation is not None:
            img = self.augmentation(img)
        img = self.m_net_transform(img)
        return img, clazz

    def __getitem__(self, index) -> T_co:
        return self.load_sample(self.samples[index])


class RPSDatamodule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        data_dir: str | Path = Path.home().joinpath('PycharmProjects', 'StudyWeek2023', 'dataset'),
        augmentation: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.datasets = None

    def setup(self, stage=None):
        splits = RPSDataset.splits_from_dir(self.data_dir)
        joined_splits = list(chain.from_iterable(splits.values()))
        assert len(joined_splits) == len(set(joined_splits))
        self.datasets = {
            k: RPSDataset(splits[k], augmentation=self.augmentation and k == 'train')
            for k in splits
        }

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class MobileNetV3RPS(LightningModule):
    def __init__(self, lr: float = 0.001):
        super().__init__()
        model = mobilenet_v3_small(MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 3)
        self.model = model
        self.lr = lr
        self.loss = CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=3)
        self.valid_acc = Accuracy(task='multiclass', num_classes=3)
        self.test_acc = Accuracy(task='multiclass', num_classes=3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, gt = batch
        logits = self.model(img)
        loss = self.loss(logits, gt)
        pred = torch.argmax(logits, dim=1)
        self.log("train_loss", loss)
        self.train_acc.update(pred, gt)
        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        logits = self.model(img)
        loss = self.loss(logits, gt)
        pred = torch.argmax(logits, dim=1)
        self.log("val_loss", loss)
        self.valid_acc.update(pred, gt)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.valid_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        img, gt = batch
        logits = self.model(img)
        pred = torch.argmax(logits, dim=1)
        self.test_acc.update(pred, gt)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
        return [optimizer], [scheduler]


def main():
    experiments_root = Path.home().joinpath('PycharmProjects', 'StudyWeek2023', 'experiments')
    experiments_root.mkdir(parents=False, exist_ok=True)
    curr_exp_root = experiments_root.joinpath(uuid4().hex)
    curr_exp_root.mkdir(parents=False, exist_ok=False)

    model = MobileNetV3RPS(lr=0.001)
    dm = RPSDatamodule(batch_size=64)
    trainer = Trainer(
        accelerator='auto',
        log_every_n_steps=1,
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(
                dirpath=curr_exp_root/'checkpoints',
                filename='{epoch}-{val_acc:.2f}',
                save_top_k=1,
                mode="max",
                monitor="val_acc",
                save_last=True,
            ),
        ],
        logger=CSVLogger(
            save_dir=curr_exp_root/'logs',
            name="mobilenet_v3_rps",
        ),
    )
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm, ckpt_path='best')


if __name__ == "__main__":
    main()
