import os
import random
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_processed_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abs_path = to_absolute_path(path)

    if abs_path.endswith(".csv"):
        df = pd.read_csv(abs_path)
    elif abs_path.endswith((".pkl", ".pickle")):
        df = joblib.load(abs_path)
    else:
        raise ValueError("Supported formats: .csv / .pkl")

    y = df["Rent"]
    X = df.drop(columns=["Rent"])

    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(
        X.values,
        y.values,
        test_size=0.2,
        random_state=365
    )

def build_model(params: Dict[str, Any], seed: int) -> Any:
    return RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
        **params
    )

def evaluate(model, X_train, y_train, X_test, y_test, metric: str) -> float:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if metric == "rmse":
        mse = mean_squared_error(y_test, preds)
    return float(np.sqrt(mse))

    if metric == "r2":
        return float(r2_score(y_test, preds))

    raise ValueError("Metric must be 'rmse' or 'r2'")

def evaluate_cv(model, X, y, metric: str, seed: int, n_splits: int = 5) -> float:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in cv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = clone(model)
        scores.append(evaluate(m, X_tr, y_tr, X_te, y_te, metric))

    return float(np.mean(scores))

def make_sampler(name: str, seed: int):
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError("Sampler must be 'tpe' or 'random'")

def suggest_params(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    space = cfg.hpo.random_forest

    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            space.n_estimators.low,
            space.n_estimators.high
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            space.max_depth.low,
            space.max_depth.high
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            space.min_samples_split.low,
            space.min_samples_split.high
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            space.min_samples_leaf.low,
            space.min_samples_leaf.high
        )
    }

def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):

            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)

            model = build_model(params, seed=cfg.seed)

            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_test])
                y = np.concatenate([y_train, y_test])
                score = evaluate_cv(
                    model, X, y,
                    metric=cfg.hpo.metric,
                    seed=cfg.seed,
                    n_splits=cfg.hpo.cv_folds
                )
            else:
                score = evaluate(
                    model,
                    X_train, y_train,
                    X_test, y_test,
                    metric=cfg.hpo.metric
                )

            mlflow.log_metric(cfg.hpo.metric, score)

            return score

    return objective

def main(cfg: DictConfig):

    set_global_seed(cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_test, y_train, y_test = load_processed_data(
        cfg.data.processed_path
    )

    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed)

    with mlflow.start_run(run_name="hpo_parent") as parent_run:

        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", cfg.seed)

        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True),
            "config_resolved.json"
        )

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler
        )

        objective = objective_factory(
            cfg, X_train, X_test, y_train, y_test
        )

        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_trial = study.best_trial

        mlflow.log_metric(
            f"best_{cfg.hpo.metric}",
            float(best_trial.value)
        )

        mlflow.log_dict(best_trial.params, "best_params.json")

        # Retrain best model
        best_model = build_model(best_trial.params, seed=cfg.seed)

        final_score = evaluate(
            best_model,
            X_train, y_train,
            X_test, y_test,
            metric=cfg.hpo.metric
        )

        mlflow.log_metric(
            f"final_{cfg.hpo.metric}",
            final_score
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, "model")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_entry()