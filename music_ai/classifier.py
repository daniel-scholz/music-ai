import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicClassifier(pl.LightningModule):
    """Class to classifiy music in WAV format."""

    def __init__(self, cfg: dict, input_size: int = 1):
        super(MusicClassifier, self).__init__()

        num_classes = cfg["num_classes"]
        seq_length = cfg["data"]["max_sequence_length"]
        hidden_size = cfg["hidden_size"]

        self.rnn1 = nn.RNN(
            input_size, hidden_size=hidden_size, batch_first=True, num_layers=1
        )
        self.fc1 = nn.Linear(hidden_size, num_classes)

        self.cfg = cfg

    def forward(self, wav_seq):
        """Predict composer for each piece of music."""
        # agree with the documentation of RNN while having better variable naming
        # encode wav sequence to hidden state
        batch_size = wav_seq.size(0)

        h_0 = torch.randn(1, batch_size, self.rnn1.hidden_size).to(wav_seq.device)
        # for LSTMs
        # c0 = torch.randn(1, batch_size, self.rnn1.hidden_size).to(wav_seq.device)
        feats, h_0 = self.rnn1(wav_seq.view(batch_size, -1, 1), h_0)
        # predict class
        logits = self.fc1(h_0.reshape(batch_size, -1))
        return logits

    def training_step(self, batch, batch_idx):
        # extract input and target from batch
        metrics = self._calc_loss_and_metrics(batch, batch_idx)
        loss = metrics["loss"]
        self._log_metrics(metrics, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = self._calc_loss_and_metrics(batch, batch_idx)
        self._log_metrics(metrics, "val")
        return metrics["loss"]

    def _log_metrics(self, metric_dict: dict, mode: str):
        for key, value in metric_dict.items():
            self.log(mode + "_" + key, value)

    def _calc_loss_and_metrics(self, batch, batch_idx):
        net_in = batch[0]
        target = batch[1].long()

        logits = self(net_in)

        loss = F.cross_entropy(logits, target)
        accuracy = self._calc_accuracy(logits, target)

        metrics = {"loss": loss, "accuracy": accuracy}

        return metrics

    def _calc_accuracy(self, logits, target):
        """Calculate accuracy based on logits and integer targets."""
        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == target).float()
        accuracy = correct.sum() / len(correct)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"])
        return optimizer

