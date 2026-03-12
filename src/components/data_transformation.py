import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "satisfaction_level",
                "last_evaluation",
                "number_project",
                "average_montly_hours",
                "time_spend_company"
            ]

            categorical_columns = [
                "Work_accident",
                "promotion_last_5years",
                "Department",
                "salary"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "left"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessing_obj = self.get_data_transformer_object()

            X_train_processed = preprocessing_obj.fit_transform(X_train)
            X_test_processed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_processed, y_train]
            test_arr = np.c_[X_test_processed, y_test]

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)