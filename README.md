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

# OpenAI Models

For OpenAI models, export your API key as an environment variable:

Linux or macOS:

```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

Or in windows:

```bash
setx OPENAI_API_KEY "your_openai_api_key_here"
```

# Llama Models via OpenRouter

To use Llama models with OpenRouter, follow these steps:

1. Visit [OpenRouter](https://openrouter.ai/) and log in or create an account.
2. Navigate to the API keys section in your account dashboard and generate a new API key.
3. Export the API key as an environment variable:
   - For Linux or macOS:
     ```bash
     export OPENROUTER_API_KEY='your_openrouter_api_key_here'
     ```
   - For Windows:
     ```bash
     setx OPENROUTER_API_KEY "your_openrouter_api_key_here"
     ```
4. Verify the environment variable:
   - On Linux or macOS:
     ```bash
     echo $OPENROUTER_API_KEY
     ```
   - On Windows:
     ```bash
     echo %OPENROUTER_API_KEY%
     ```


## Run the App as CLI with Options

You can run the `hs_optimize` command-line interface (CLI) with several options for customizing the optimization process.
Make sure the joblib data file contains a python dictionary with the keys 'X_train', 'y_train', 'X_test', and 'y_test'.
The application uses 'y_train' data to determine if it is a classification or regression problem.

### Usage

hs_optimize [-h] --data DATA [--history-file-path HISTORY_FILE_PATH] [--model MODEL] [--iterations ITERATIONS]


### Optional Arguments

- **`-h, --help`**:  
  Show the help message and exit.

- **`--data DATA`, `-d DATA`**:  
  Path to a `.joblib` file containing training and test data. The file should include a dictionary with keys like `'X_train'`, `'y_train'`, `'X_test'`, and `'y_test'`. These should be NumPy arrays representing the feature and target datasets for model training and evaluation.

- **`--history-file-path HISTORY_FILE_PATH`, `-hfp HISTORY_FILE_PATH`**:  
  Path to the `.txt`  or `.joblib` file where the model history will be saved. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is `'model_history.joblib'`.

- **`--model MODEL`, `-m MODEL`**:  
  The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to `'meta-llama/llama-3.1-405b-instruct:free'`.

- **`--iterations ITERATIONS`, `-i ITERATIONS`**:  
  The number of iterations to run. Each iteration involves training a model, evaluating its performance, and generating improvements. Default is `5`.


- **`--extra-info EXTRA_INFO, -ei EXTRA_INFO`**:
   Additional context or information to provide to the LLM for more informed suggestions. Examples include class imbalance, noisy labels, or outlier data. Default is 'Not available'.

### Example 1

Here’s an example of how to run the app with custom data, model history path, and iterations:

```bash
hs_optimize -d my_classification_data.joblib -hfp output_model_history.joblib -i 10 -m gpt-4o
```

### Example 2

Example with Class Imbalance for Classification:

```bash
hs_optimize -d my_classification_data.joblib -hfp classification_history.txt -i 10 --extra-info "Binary classification with class imbalance, 4:1 ratio between class 0 and class 1."
```

In this case, the application will pass the additional information to the LLM, which can then suggest using custom loss functions or class weighting techniques to address the class imbalance.

## License
[MIT](LICENSE)

