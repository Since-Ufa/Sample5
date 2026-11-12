"""
Распознавание рукописных цифр (0–9) на PyTorch Lightning
с автоматическим логированием в MLflow.
"""
import lightning as L
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("digits-autolog")
mlflow.pytorch.autolog()  # дальше MLflow сам записывает параметры, метрики и модель


def load_data():
    digits = load_digits()  # 1797 картинок цифр 8x8
    X = torch.tensor(digits.data / 16.0, dtype=torch.float32)  # 64 пикселя, яркость от 0 до 1
    y = torch.tensor(digits.target)  # правильный ответ: какая это цифра (0..9)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # DataLoader подаёт данные в модель порциями (батчами) по 64 картинки
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)
    return train_loader, test_loader


class DigitsModel(L.LightningModule):
    def __init__(self, hidden, lr):
        super().__init__()
        self.lr = lr
        # 64 пикселя на входе -> скрытый слой из `hidden` нейронов -> 10 выходов (по одному на каждую цифру)
        self.net = nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)  # насколько модель ошибается
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.loss_fn(logits, y))
        self.log("val_accuracy", (logits.argmax(1) == y).float().mean())  # доля угаданных цифр

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(lr, epochs, hidden):
    L.seed_everything(0)
    train_loader, test_loader = load_data()
    model = DigitsModel(hidden=hidden, lr=lr)
    trainer = L.Trainer(max_epochs=epochs, enable_progress_bar=False)

    with mlflow.start_run(run_name=f"lr={lr}, hidden={hidden}"):
        trainer.fit(model, train_loader, test_loader)  # обучение; всё логируется автоматически


if __name__ == "__main__":
    train(lr=0.01, epochs=10, hidden=32)
    train(lr=0.1, epochs=10, hidden=64)
    train(lr=0.001, epochs=20, hidden=32)
