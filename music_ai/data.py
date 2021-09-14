import pathlib

import pandas as pd
import torch
from scipy.io import wavfile
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(0)


class WAVDataset(torch.utils.data.Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.fns = self._load_fns(config["data"]["raw_data_path"])
        self.labels = self._load_labels(config["data"]["label_path"])
        if config["overfit"]:
            n_overfit = 10
            self.labels = self.labels[:n_overfit]
            self.fns = self.fns[:n_overfit]

        self.max_sequence_length = config["data"]["max_sequence_length"]

    def __len__(self) -> int:
        return len(self.fns)

    def __getitem__(self, index) -> torch.Tensor:
        """"""
        sample_rate, raw_wav_data = self._read_wav(self.fns[index])
        label = self.labels[index]
    
        # crop wave data to max sequence length
        cropped_wav_data = self._crop_sequence(raw_wav_data)
        # print(f"{raw_wav_data=}, {samplerate=}, {label=}")

        return cropped_wav_data, label

    def _crop_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Crop a sequence to a fixed length."""
        return seq[: min(self.max_sequence_length, len(seq))]

    def _load_fns(self, raw_data_path: str) -> list:
        """Loads the list of filenames."""
        raw_data_path = pathlib.Path(raw_data_path)
        return list(raw_data_path.glob("*.wav"))

    def _read_wav(self, fn: pathlib.Path) -> torch.Tensor:
        """Reads a wav file into a tensor."""
        samplerate, data = wavfile.read(fn)
        return samplerate, torch.from_numpy(data).float()

    def _load_labels(self, label_path: str) -> torch.Tensor:
        """Loads the list of labels."""
        label_path = pathlib.Path(label_path)
        # read csv file from label path
        with open(label_path, "r") as f:
            labels = pd.read_csv(f)["composer"]

        # labels as one hot encoding
        labels_int = labels.astype("category").cat.codes

        # log current mapping
        mapping = pd.concat([labels, labels_int], axis=1).drop_duplicates()
        mapping.to_csv(label_path.parent / "composer_int_mapping.csv", index=False)

        return torch.tensor(labels_int.values)

    def collate_fn(self, batch):
        """Pad sequences."""

        data = pad_sequence([b[0] for b in batch], batch_first=True)
        labels = torch.stack([b[1] for b in batch], dim=0)
        return data, labels


class ComposerDataset(WAVDataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> torch.Tensor:
        """Returns audio and composer as label"""
        return super().__getitem__(index)

