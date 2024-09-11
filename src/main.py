from src.data_handler import DataHandler
from src.model_trainer import ModelTrainer
from src.model_selector import ModelSelector
from src.llm_improver import LLMImprover
from src.hyperparameter_tuner import HyperparameterTuner
from src.experiment_manager import ExperimentManager
from google.generativeai import GenerativeModel

def main():
    # Initialize components
    data_handler = DataHandler()
    model_trainer = ModelTrainer()
    model_selector = ModelSelector()
    llm_improver = LLMImprover(GenerativeModel())
    hyperparameter_tuner = HyperparameterTuner()

    # Experiment Manager
    experiment_manager = ExperimentManager(
        data_handler, 
        model_trainer, 
        model_selector, 
        llm_improver, 
        hyperparameter_tuner,
        max_iterations=5
    )

    # Run the experiment
    best_model = experiment_manager.run_experiment(filepath='path/to/your/joblib/file')

    print("Best model trained successfully!")

if __name__ == "__main__":
    main()


