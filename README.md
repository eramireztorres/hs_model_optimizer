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

## Run the App as CLI with Options

You can run the `hs_optimize` command-line interface (CLI) with several options for customizing the optimization process.

### Usage

hs_optimize [-h] --data DATA [--history-file-path HISTORY_FILE_PATH] [--model MODEL] [--iterations ITERATIONS]


### Optional Arguments

- **`-h, --help`**:  
  Show the help message and exit.

- **`--data DATA`, `-d DATA`**:  
  Path to a `.joblib` file containing training and test data. The file should include a dictionary with keys like `'X_train'`, `'y_train'`, `'X_test'`, and `'y_test'`. These should be NumPy arrays representing the feature and target datasets for model training and evaluation.

- **`--history-file-path HISTORY_FILE_PATH`, `-hfp HISTORY_FILE_PATH`**:  
  Path to the `.joblib` file where the model history will be saved. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is `'model_history.joblib'`.

- **`--model MODEL`, `-m MODEL`**:  
  The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to `'gpt-4o'`.

- **`--iterations ITERATIONS`, `-i ITERATIONS`**:  
  The number of iterations to run. Each iteration involves training a model, evaluating its performance, and generating improvements. Default is `5`.

### Example

Here’s an example of how to run the app with custom data, model history path, and iterations:

```bash
hs_optimize my_classification_data.joblib -hfp output_model_history.joblib -i 4
```


## License
[MIT](LICENSE)

