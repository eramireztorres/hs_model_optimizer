import pandas as pd
import os
import warnings
import joblib
from typing import Optional, List, Dict, Any, Callable
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, agent_tool
from google.adk.models.lite_llm import LiteLlm
import subprocess
from google.adk.agents import Agent


def fix_csv_target_column(
    input_csv_path: str,
    target_column: Optional[str] = None,
    output_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ensure the target column is the last column for hs_optimize.

    Steps:
    1. Load the CSV from `input_csv_path`.
    2. Determine `target_column`: if `None`, use the last column; if an integer string, interpret as column index.
    3. Reorder columns so that the target is last.
    4. Write the fixed CSV to `output_csv_path` (overwrites original if None).
    5. Return metadata including row count, feature names, and column types.
    """
    # Load CSV
    df = pd.read_csv(input_csv_path)

    # Infer target column
    if target_column is None:
        tc = df.columns[-1]
    else:
        try:
            idx = int(target_column)
            tc = df.columns[idx]
        except (ValueError, KeyError, IndexError):
            tc = str(target_column)

    if tc not in df.columns:
        warnings.warn(
            f"Target column '{tc}' not found in CSV columns {list(df.columns)}. "
            f"Defaulting to last column '{df.columns[-1]}'."
        )
        tc = df.columns[-1]

    # Reorder columns: features first, then target
    features = [c for c in df.columns if c != tc]
    ordered_cols = features + [tc]
    df = df[ordered_cols]

    # Write output
    if output_csv_path is None:
        output_csv_path = input_csv_path
    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # Prepare metadata
    metadata: Dict[str, Any] = {
        "fixed_csv": output_csv_path,
        "n_rows": int(df.shape[0]),
        "n_features": int(len(features)),
        "feature_columns": features,
        "target_column": tc,
        "column_types": {col: str(df[col].dtype) for col in df.columns},
    }
    return metadata


# Register as a FunctionTool for the root agent
def fix_csv_target_column_tool():
    return FunctionTool(func=fix_csv_target_column)



def split_dataset(
    input_path: str,
    partition_specs: List[Dict[str, Any]],
    target_column: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Split an input dataset (CSV or joblib) into train/test sets based on partition_specs.

    Parameters:
    - input_path: path to input .csv or .joblib file containing a DataFrame with the target as last column or named.
    - partition_specs: list of conditions dicts with keys:
        - 'column': column name to filter on
        - 'operator': one of ['==','!=','<','>','<=','>=','in','notin']
        - 'value': value or list of values for comparison
      Rows satisfying ALL specs become the TEST set.
    - target_column: optional; if None, use last column as target.
    - output_dir: directory to write outputs; defaults to input file's directory.
    - output_format: 'csv' or 'joblib'; if None, inferred from input_path.

    Returns metadata with file paths and record counts.
    """
    # Load DataFrame
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.joblib':
        df = joblib.load(input_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Loaded object from {input_path} is not a pandas DataFrame.")
    else:
        df = pd.read_csv(input_path)

    # Determine output format
    fmt = output_format or ( 'joblib' if ext == '.joblib' else 'csv' )
    # Determine output directory
    base_dir = output_dir or os.path.dirname(input_path) or '.'
    os.makedirs(base_dir, exist_ok=True)

    # Determine target column
    if target_column is None:
        tc = df.columns[-1]
    else:
        if target_column in df.columns:
            tc = target_column
        else:
            warnings.warn(f"Target column '{target_column}' not found; defaulting to last column '{df.columns[-1]}'")
            tc = df.columns[-1]

    # Build mask for TEST set
    ops: Dict[str, Callable[[pd.Series, Any], pd.Series]] = {
        '==': lambda s, v: s == v,
        '!=': lambda s, v: s != v,
        '<':  lambda s, v: s < v,
        '>':  lambda s, v: s > v,
        '<=': lambda s, v: s <= v,
        '>=': lambda s, v: s >= v,
        'in': lambda s, v: s.isin(v if isinstance(v, (list, tuple, set)) else [v]),
        'notin': lambda s, v: ~s.isin(v if isinstance(v, (list, tuple, set)) else [v])
    }
    mask = pd.Series(True, index=df.index)
    for spec in partition_specs:
        col = spec.get('column')
        op = spec.get('operator')
        val = spec.get('value')
        if col not in df.columns:
            warnings.warn(f"Partition spec column '{col}' not in DataFrame; skipping this condition.")
            continue
        if op not in ops:
            warnings.warn(f"Unsupported operator '{op}' for column '{col}'; skipping this condition.")
            continue
        try:
            mask &= ops[op](df[col], val)
        except Exception as e:
            warnings.warn(f"Error applying operator '{op}' on column '{col}': {e}; skipping this condition.")

    df_test = df[mask].reset_index(drop=True)
    df_train = df[~mask].reset_index(drop=True)

    # Split features/target
    features = [c for c in df.columns if c != tc]
    X_train, y_train = df_train[features], df_train[tc]
    X_test,  y_test  = df_test[features],  df_test[tc]

    # Define output paths
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    paths = {}
    for name, obj in [('X_train', X_train), ('y_train', y_train), ('X_test', X_test), ('y_test', y_test)]:
        file_name = f"{base_name}_{name}.{ 'joblib' if fmt=='joblib' else 'csv' }"
        full_path = os.path.join(base_dir, file_name)
        if fmt == 'joblib':
            joblib.dump(obj, full_path)
        else:
            obj.to_csv(full_path, index=False)
        paths[name] = full_path

    metadata: Dict[str, Any] = {
        'paths': paths,
        'n_rows_train': int(df_train.shape[0]),
        'n_rows_test': int(df_test.shape[0]),
        'partition_specs': partition_specs,
        'n_features': len(features),
        'feature_columns': features,
        'target_column': tc
    }
    return metadata

# Register as a FunctionTool
def split_dataset_tool():
    return FunctionTool(func=split_dataset)


def read_model_history(
    history_path: str,
    index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Read a hs_optimize .joblib history file and optionally fetch a specific entry.

    - If index is None, returns a summary listing number of entries and global metrics for each.
    - If index is provided, returns the model_code and metrics dict for that entry.

    Does not raise errors but emits warnings and returns best-effort defaults.

    Returns:
      {
        'n_models': int,
        'summaries': List[Dict[str, Any]]  # only if index is None
        'model_code': str,                 # only if index provided
        'metrics': Dict[str, Any]          # only if index provided
      }
    """
    if not os.path.exists(history_path):
        warnings.warn(f"History file '{history_path}' not found.")
        return {'n_models': 0, 'summaries': [] if index is None else None}

    try:
        history = joblib.load(history_path)
    except Exception as e:
        warnings.warn(f"Failed to load joblib file '{history_path}': {e}")
        return {'n_models': 0, 'summaries': [] if index is None else None}

    if not isinstance(history, list):
        warnings.warn(f"Expected a list of model entries, got {type(history).__name__}.")
        return {'n_models': 0, 'summaries': [] if index is None else None}

    n = len(history)
    result: Dict[str, Any] = {'n_models': n}

    # Summaries: global_metrics for each entry
    if index is None:
        summaries: List[Dict[str, Any]] = []
        for i, entry in enumerate(history):
            if not isinstance(entry, dict):
                warnings.warn(f"Entry {i} is not a dict; skipping.")
                continue
            metrics = entry.get('metrics', {})
            global_metrics = metrics.get('global_metrics', metrics)
            summaries.append({'index': i, 'global_metrics': global_metrics})
        result['summaries'] = summaries
        return result

    # Specific entry
    if not isinstance(index, int) or index < 0 or index >= n:
        warnings.warn(f"Index {index} out of range [0, {n-1}]. Defaulting to 0.")
        index = 0

    entry = history[index]
    if not isinstance(entry, dict):
        warnings.warn(f"Entry at index {index} is not a dict. Returning empty structure.")
        return {'n_models': n, 'model_code': None, 'metrics': None}

    model_code = entry.get('model_code')
    if model_code is None:
        warnings.warn(f"No 'model_code' found in entry {index}.")

    metrics = entry.get('metrics')
    if metrics is None:
        warnings.warn(f"No 'metrics' found in entry {index}.")

    result['model_code'] = model_code
    result['metrics'] = metrics
    return result


# Register as a FunctionTool for the root agent
def read_model_history_tool():
    return FunctionTool(func=read_model_history)


# Default LLM for code generation
DEFAULT_MODEL = LiteLlm(model="openai/gpt-4.1-mini")

# --- SUB-AGENT: Model Code Generator ---
# Generates Python code files defining a load_model function per CLI contract
model_code_generator = LlmAgent(
    name="ModelCodeGenerator",
    model=DEFAULT_MODEL,
    instruction="""
You are a Model Code Generator. Given a summary of desired model characteristics and dataset context, write a complete Python file that defines a `load_model` function. Follow these rules exactly:
1. For scikit-learn models:
   - Signature: `def load_model():`
   - Inside, import the needed estimator and return an untrained instance with default or specified hyperparameters.
2. For neural networks:
   - Signature: `def load_model(X_train, y_train, **kwargs):`
   - Include imports, reproducibility seed, define a PyTorch `nn.Module`, and return `model, optimizer, criterion`.
3. Do not include any top-level code other than imports inside `load_model`.
4. Output only the code (no commentary or markdown).
""",
    description="LLM-based generator of `load_model` code for hs_optimizer",
    output_key="generated_code",
)
# Wrap as a tool for the root agent to invoke
code_generator_tool = agent_tool.AgentTool(agent=model_code_generator)

# --- TOOL: Save Generated Model Code to File ---

def save_model_code(file_path: str, code: str) -> Dict[str, str]:
    """
    Write the provided Python `code` to `file_path` and return the file path.
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(code)
    return {"file_path": file_path}

save_model_code_tool = FunctionTool(func=save_model_code)




def run_hs_optimize(
    data: str,
    model: Optional[str] = None,
    model_provider: Optional[str] = None,
    history_file_path: Optional[str] = None,
    iterations: Optional[int] = None,
    extra_info: Optional[str] = None,
    output_models_path: Optional[str] = None,
    is_regression: Optional[str] = None,
    metrics_source: Optional[str] = None,
    error_model: Optional[str] = None,
    quiet: bool = True
) -> Dict[str, Any]:
    """
    Invoke the hs_optimize CLI with provided parameters.

    Builds the command, runs subprocess, and captures output.
    Emits warnings on non-zero exit codes but does not raise.

    Returns:
      {
        'exit_code': int,
        'stdout': str,
        'stderr': str,
        'history_file_path': str  # echo of provided path or default
      }
    """
    # Base command
    cmd = ['hs_optimize', '--data', data]

    # Optional arguments
    if model is not None:
        cmd += ['--model', model]
    if model_provider is not None:
        cmd += ['--model-provider', model_provider]
    if history_file_path is not None:
        cmd += ['--history-file-path', history_file_path]
    if iterations is not None:
        cmd += ['--iterations', str(iterations)]
    if extra_info is not None:
        cmd += ['--extra-info', extra_info]
    if output_models_path is not None:
        cmd += ['--output-models-path', output_models_path]
    if is_regression is not None:
        cmd += ['--is-regression', is_regression]
    if metrics_source is not None:
        cmd += ['--metrics-source', metrics_source]
    if error_model is not None:
        cmd += ['--error-model', error_model]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
    except FileNotFoundError:
        warnings.warn("`hs_optimize` CLI not found in PATH.")
        return {'exit_code': -1, 'stdout': '', 'stderr': 'CLI not found', 'history_file_path': history_file_path}
    except Exception as e:
        warnings.warn(f"Failed to run hs_optimize: {e}")
        return {'exit_code': -1, 'stdout': '', 'stderr': str(e), 'history_file_path': history_file_path}

    if proc.returncode != 0:
        warnings.warn(
            f"hs_optimize exited with code {proc.returncode}. stderr: {proc.stderr}"
        )

    return {
        'exit_code': proc.returncode,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
        'history_file_path': history_file_path
    }


# Register as a FunctionTool for the root agent
def run_hs_optimize_tool():
    return FunctionTool(func=run_hs_optimize)

root_agent = Agent(
    name="HsOptimizeCoordinator",
    model=DEFAULT_MODEL,
    description=(
        "Orchestrates the full hs_optimize workflow: "
        "preprocess input, split data if requested, run optimization, "
        "analyze results, and optionally generate new model code."
    ),
    instruction="""
You are the HsOptimizeCoordinator. Given a user request to optimize a model, follow these steps:

1. **Ensure input is ready**  
   - If the user provided a single CSV, call `fix_csv_target_column` to reorder the target to the last column.  
   - If the user provided pre-split Joblib or CSVs, skip this.

2. **Optional custom split**  
   - If the user specified `partition_specs`, invoke `split_dataset` with their specs to produce `X_train`, `y_train`, `X_test`, `y_test`.  
   - Otherwise, the CLI will handle splitting internally.

3. **Run optimization**  
   - Call `run_hs_optimize` with the appropriate `--data` argument (the fixed CSV, split-joblib, or folder), plus model, iterations, extra-info, etc., based on user flags.

4. **Inspect history**  
   - Call `read_model_history` with no index to get a summary of `global_metrics`.  
   - If the best metric meets the user’s goal (e.g., accuracy ≥ threshold), report success.

5. **Iterate if needed**  
   - If no entry is satisfactory, ask the user for desired changes or automatically invoke `code_generator_tool` to produce new `load_model()` code.  
   - Save it via `save_model_code_tool`, then call `run_hs_optimize` again on that code.

6. **Return**  
   - Provide the final `history_file_path`, key metrics, and any generated model code file path.

Use these tools if needed. Always confirm critical arguments before invoking a tool.
""",
    tools=[
        fix_csv_target_column_tool(),
        split_dataset_tool(),
        run_hs_optimize_tool(),
        read_model_history_tool(),
        code_generator_tool,
        save_model_code_tool,
    ],
)

