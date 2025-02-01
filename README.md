# LLM-Powered Model Optimizer Project

This project aims to optimize machine learning models by iterating through model training, evaluation, and improvements using an LLM (Large Language Model). The system dynamically improves models and hyperparameters across scikit-learn, LightGBM and XGBoost models.

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

### OpenAI Models

For OpenAI models, export your API key as an environment variable:

Linux or macOS:

```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

Or in windows:

```bash
setx OPENAI_API_KEY "your_openai_api_key_here"
```

### Other Models via OpenRouter

To use Llama, Gemini or other models with OpenRouter, follow these steps:

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

## Run the App as CLI with Options

You can run the `hs_optimize` command-line interface (CLI) with several options for customizing the optimization process.  
The application supports various input data formats and automatically determines whether the problem is classification or regression based on the target (`y`) values.

### Usage

```bash
hs_optimize [-h] --data DATA [--history-file-path HISTORY_FILE_PATH] [--model MODEL] [--iterations ITERATIONS]
```

### Supported Input File Formats

The application supports the following input formats:

1. **Pre-split `.joblib` file**:  
   A Python dictionary containing the keys:  
   - `'X_train'`, `'y_train'` (training data),  
   - `'X_test'`, `'y_test'` (test data).

2. **Pre-split `.csv` files**:  
   A directory containing the following files:  
   - `X_train.csv`, `y_train.csv`, `X_test.csv`, and `y_test.csv`.

3. **Unsplit `.joblib` file**:  
   A Python dictionary containing the keys:  
   - `'X'` (features),  
   - `'y'` (targets).  
   The application will create a validation split from the data (default split ratio is 80/20).

4. **Unsplit `.csv` files**:  
   A directory containing two files:  
   - `X.csv` (features),  
   - `y.csv` (targets).  
   The application will create a validation split.

5. **Single `.csv` file**:  
   A single CSV file where:  
   - All columns except the last are treated as features (`X`),  
   - The last column is assumed to be the target (`y`).



### Arguments

- **`-h, --help`**:  
  Show the help message and exit.

- **`--data DATA`, `-d DATA`**:  
  Path to the input dataset. Supported formats are .joblib files, directories containing .csv files, or a single .csv file.
  The application handles validation splits automatically for unsplit datasets.

- **`--history-file-path HISTORY_FILE_PATH`, `-hfp HISTORY_FILE_PATH`**:  
  Path to the `.txt`  or `.joblib` file where the model history will be saved. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is `'model_history.joblib'`.

- **`--model MODEL`, `-m MODEL`**:  
  The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. 
  Examples: `'meta-llama/llama-3.2-3b-instruct:free'`, `'deepseek/deepseek-chat'`. Defaults to `'gpt-4o-mini'`.

- **`--is-regression IS_REGRESSION`, `-ir IS_REGRESSION`**:
  Specify the type of model to train.  
  Options:
    - `true`: Regression.
    - `false`: Classification.
    If not specified, the model type is inferred from the data target: if they are all integers, it is assumed to be classification.

- **`--metrics-source METRICS_SOURCE`, `-ms METRICS_SOURCE`**:  
  Specify the source of the metrics to show to the LLM.  
  Options:
    - `validation` (default): Metrics are computed on a validation split created from the training data.
    - `test`: Metrics are computed on the test data.

- **`--iterations ITERATIONS`, `-i ITERATIONS`**:  
  The number of iterations to run. Each iteration involves training a model, evaluating its performance, and generating improvements. Default is `10`.

- **`--error-model ERROR_MODEL`, `-em ERROR_MODEL`**:  
  The name of the LLM model to use for error corrections. If not specified, `--model` is used.

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

