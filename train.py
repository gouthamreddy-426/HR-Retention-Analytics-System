from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    
    pipeline = TrainPipeline()
    
    print("Starting Training Pipeline...")
    
    pipeline.run_pipeline()
    
    print("Training Completed Successfully")