# Model Optimizer Project

This project aims to optimize machine learning models by iterating through model training, evaluation, and improvements using an LLM (Large Language Model). The system dynamically improves models and hyperparameters across scikit-learn and XGBoost classifiers.

## Features
- Supports multiple classifiers.
- Utilizes LLM to suggest code and hyperparameter improvements.
- Dynamically applies improvements using hot-swapping techniques.
- Saves model history and evaluation metrics.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/my_model_optimizer_project.git
    cd my_model_optimizer_project
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the application:
    ```
    python main.py
    ```

## Project Structure
- `src/`: Contains core classes and logic.
- `tests/`: Unit tests for the project.
- `logs/`: Contains logs for tracking experiments.

## License
[MIT](LICENSE)

