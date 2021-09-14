"""Module containing training procedure."""


import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from music_ai.classifier import MusicClassifier
from music_ai.data import WAVDataset

torch.manual_seed(0)


def train(cfg: dict):
    """Train the model."""
    print("Training the model...")

    wav_ds = WAVDataset(cfg)
    dataset_lengths = [int(0.7 * len(wav_ds)), int(0.2 * len(wav_ds))]
    train_ds, val_ds, test_ds = random_split(
        wav_ds, [*dataset_lengths, len(wav_ds) - sum(dataset_lengths)],
    )

    def dataloader(ds: torch.utils.data.Dataset, shuffle: bool) -> DataLoader:
        """Return dataloader object with preset params."""
        return DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=shuffle,
            collate_fn=wav_ds.collate_fn,
            pin_memory=True,
            num_workers=4,
            persistent_workers=False,
        )

    train_dl = dataloader(train_ds, shuffle=True)
    val_dl = dataloader(val_ds, shuffle=False)

    classifier = MusicClassifier(cfg=cfg)

    logger = TensorBoardLogger(cfg["log_dir"], name="music_classifier")
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=1,  # min(len(train_ds) - 1, 50),
        gpus=1,
        auto_scale_batch_size="binsearch",
    )
    trainer.fit(classifier, train_dl, val_dl)
