import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def plot_feature_importance(model, feature_names, output_path="feature_importance.png"):
    importances = model.feature_importances_

    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
