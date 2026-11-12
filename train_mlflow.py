import mlflow
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # куда отправлять логи (запущенный `mlflow ui`)
mlflow.set_experiment("digits")  # эксперимент = папка для запусков; создастся сам


def load_data():
    digits = load_digits()  # 1797 картинок цифр 8x8
    X = torch.tensor(digits.data / 16.0, dtype=torch.float32)  # 64 пикселя, яркость от 0 до 1
    y = torch.tensor(digits.target)  # правильный ответ: какая это цифра (0..9)
    # 80% картинок — для обучения, 20% — для проверки на невиданных примерах
    return train_test_split(X, y, test_size=0.2, random_state=42)


def make_model(hidden):
    # 64 пикселя на входе -> скрытый слой из `hidden` нейронов -> 10 выходов (по одному на каждую цифру)
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def accuracy(model, X, y):
    # доля картинок, на которых модель угадала цифру
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def train(lr, epochs, hidden):
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = load_data()
    model = make_model(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=f"lr={lr}, hidden={hidden}"):  # один run = один запуск обучения
        # параметры: настройки, записываем один раз
        mlflow.log_params({"lr": lr, "epochs": epochs, "hidden": hidden})

        for epoch in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(X_train), y_train)  # насколько модель ошибается
            loss.backward()
            opt.step()  # модель чуть-чуть исправляется
            # метрика по эпохам: step=epoch -> MLflow нарисует график
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # итоговая метрика
        acc = accuracy(model, X_test, y_test)
        mlflow.log_metric("test_accuracy", acc)
        print(f"lr={lr}, hidden={hidden}: test accuracy = {acc:.3f}")


if __name__ == "__main__":
    train(lr=0.01, epochs=30, hidden=32)
    train(lr=0.1, epochs=30, hidden=64)
    train(lr=0.001, epochs=100, hidden=32)
