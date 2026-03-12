import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # Data Ingestion
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")

            # Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )

            logging.info("Data transformation completed")

            # Model Training
            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info("Model training completed")
            logging.info(f"Best model F1 Score: {score}")

        except Exception as e:
            raise CustomException(e, sys)