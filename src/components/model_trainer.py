import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and testing data")

            # Correct splitting for NumPy arrays
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            best_model = None
            best_f1 = 0

            mlflow.set_experiment("HR_Retention_Experiment")

            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    logging.info(f"{model_name} trained")

                    mlflow.log_param("model_name", model_name)

                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)

                    mlflow.sklearn.log_model(model, model_name)

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)

                    plt.figure(figsize=(6,4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.title(f"{model_name} Confusion Matrix")

                    plot_path = f"{model_name}_confusion_matrix.png"
                    plt.savefig(plot_path)

                    mlflow.log_artifact(plot_path)

                    plt.close()

                    # Best model selection
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved")

            return best_f1

        except Exception as e:
            raise CustomException(e, sys)