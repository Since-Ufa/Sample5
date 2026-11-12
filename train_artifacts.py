"""
Распознавание рукописных цифр (0–9) по картинкам 8x8 пикселей
с логированием параметров, метрик и артефактов в MLflow.
"""
import matplotlib
matplotlib.use("Agg")  # рисуем картинки в файл, без открытия окон
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
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


def plot_loss(losses):
    # график ошибки по эпохам
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("Обучение модели")
    return fig


def plot_predictions(model, X, y, n=12):
    # первые n картинок из тестовой выборки с ответом модели; ошибки — красным
    with torch.no_grad():
        preds = model(X[:n]).argmax(1)
    fig, axes = plt.subplots(2, n // 2, figsize=(n, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(8, 8), cmap="gray_r")
        correct = preds[i] == y[i]
        ax.set_title(f"модель: {preds[i].item()}", color="green" if correct else "red")
        ax.axis("off")
    return fig


def train(lr, epochs, hidden):
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = load_data()
    model = make_model(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=f"lr={lr}, hidden={hidden}"):  # один run = один запуск обучения
        # параметры: настройки, записываем один раз
        mlflow.log_params({"lr": lr, "epochs": epochs, "hidden": hidden})

        losses = []
        for epoch in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(X_train), y_train)  # насколько модель ошибается
            loss.backward()
            opt.step()  # модель чуть-чуть исправляется
            losses.append(loss.item())
            # метрика по эпохам: step=epoch -> MLflow нарисует график
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # итоговая метрика
        acc = accuracy(model, X_test, y_test)
        mlflow.log_metric("test_accuracy", acc)

        # артефакт: сохраняем график в файл и отправляем файл в MLflow
        fig = plot_loss(losses)
        fig.savefig("loss.png")
        plt.close(fig)
        mlflow.log_artifact("loss.png")

        # артефакт: картинку можно отправить и сразу, без сохранения на диск
        fig = plot_predictions(model, X_test, y_test)
        mlflow.log_figure(fig, "predictions.png")
        plt.close(fig)

        # артефакт: сама обученная модель
        mlflow.pytorch.log_model(
	     model,
             name="model",
             input_example=X_test[:5],
             serialization_format="pickle"
        )

        print(f"lr={lr}, hidden={hidden}: test accuracy = {acc:.3f}")


if __name__ == "__main__":
    train(lr=0.01, epochs=30, hidden=32)
    train(lr=0.1, epochs=30, hidden=64)
    train(lr=0.001, epochs=100, hidden=32)
