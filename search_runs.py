import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# все запуски эксперимента digits, лучшие сверху
runs = mlflow.search_runs(
    experiment_names=["digits"],
    order_by=["metrics.test_accuracy DESC"],
)
print(runs[["tags.mlflow.runName", "params.lr", "params.hidden", "metrics.test_accuracy"]])

# только запуски с hidden=32 и точностью выше 90%
good = mlflow.search_runs(
    experiment_names=["digits"],
    filter_string="params.hidden = '32' and metrics.test_accuracy > 0.9",
)
print(good[["tags.mlflow.runName", "metrics.test_accuracy"]])

# ID лучшего запуска
best = runs.iloc[0]
print("Лучший run:", best["tags.mlflow.runName"], best["run_id"])
