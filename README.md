# Model Optimizer Project

This project aims to optimize machine learning models by iterating through model training, evaluation, and improvements using an LLM (Large Language Model). The system dynamically improves models and hyperparameters across scikit-learn and XGBoost models.

## Features
- Supports multiple models.
- Utilizes LLM to suggest code and hyperparameter improvements.
- Dynamically applies improvements using hot-swapping techniques.
- Saves model history and evaluation metrics.

## Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/eramireztorres/hs_model_optimizer.git
    cd hs_model_optimizer
    ```
    
2. **Set up a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages:**
    Run the following command to install dependencies:
    ```bash
    pip install .
    ```

## Export the API keys of your models

Before using, make sure to export your OpenAI API key as an environment variable. 

Linux or macOS:

```bash
export OPENAI_API_KEY='your_api_key_here'
```

Or in windows:

```bash
set OPENAI_API_KEY=your_api_key_here
```

## Run the app as CLI with options

usage: hs_optimize [-h] --data DATA [--history-file-path HISTORY_FILE_PATH] [--model MODEL] [--iterations ITERATIONS]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA, -d DATA  A dictionary containing training and test data, with keys such as 'X_train', 'y_train', 'X_test', 'y_test'. These should be NumPy arrays representing the feature and target datasets for
                        model training and evaluation.
  --history-file-path HISTORY_FILE_PATH, -hfp HISTORY_FILE_PATH
                        Path to the joblib file where the model history will be stored. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is
                        'model_history.joblib'. (default: model_history.joblib)
  --model MODEL, -m MODEL
                        The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to 'gpt-4o'. (default: gpt-4o)
  --iterations ITERATIONS, -i ITERATIONS
                        The number of iterations to run, where each iteration involves training a model, evaluating its performance, and generating improvements. Default is 5. (default: 5)


Example:

    ```bash
    hs_optimize my_classification_data.joblib -hfp output_model_history.joblib -i 4

    ```


## License
[MIT](LICENSE)

