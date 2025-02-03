import streamlit as st
import os
import subprocess
import time
import tempfile

def set_env_variable(key, value):
    os.environ[key] = value  # Sets in the current session (works across OS)

    # Persistent setting depending on OS
    if os.name == 'nt':  # Windows
        os.system(f'setx {key} "{value}"')
    else:  # Unix-based (Linux/Mac)
        bashrc_path = os.path.expanduser('~/.bashrc')
        with open(bashrc_path, 'a') as f:
            f.write(f'\nexport {key}="{value}"\n')

# Load existing environment variables if they exist
existing_openai_key = os.getenv("OPENAI_API_KEY", "")
existing_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")


st.set_page_config(
    page_title="HS Model Optimizer",
    page_icon="🚀",  # Example: rocket emoji as favicon
    layout="wide"
)


# Sidebar for API Key settings
st.sidebar.header("⚙️ API Key Settings")

# Input for OpenAI API Key (pre-populated if already set)
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    value=existing_openai_key if existing_openai_key else "",
    type="password"
)

# Input for OpenRouter API Key (pre-populated if already set)
openrouter_api_key = st.sidebar.text_input(
    "Enter your OpenRouter API Key:",
    value=existing_openrouter_key if existing_openrouter_key else "",
    type="password"
)

# Button to save API keys
if st.sidebar.button("Save API Keys"):
    if openai_api_key:
        set_env_variable("OPENAI_API_KEY", openai_api_key)
        st.sidebar.success("OpenAI API Key saved successfully!")

    if openrouter_api_key:
        set_env_variable("OPENROUTER_API_KEY", openrouter_api_key)
        st.sidebar.success("OpenRouter API Key saved successfully!")

    if not openai_api_key and not openrouter_api_key:
        st.sidebar.warning("Please enter at least one API key to save.")


# Function to run CLI with arguments and display output
def run_cli_with_output(data_path, args_dict, output_placeholder):
    # Build the CLI command
    command = ['hs_optimize', '--data', data_path]

    for key, value in args_dict.items():
        if value is not None and value != "None":  # Skip None values
            cli_key = f'--{key.replace("_", "-")}'
            command.append(cli_key)
            command.append(str(value))

    # Display the command for debugging (optional)
    output_placeholder.code(' '.join(command), language='bash')

    # Run the command and capture real-time output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        output_placeholder.text(output)  # Update the output panel in real-time
        time.sleep(0.1)

    process.stdout.close()
    process.wait()

    # Final output after process finishes
    output_placeholder.text(output)
    st.success("Optimization completed!")




# # Function to construct full paths for history and output models
# def construct_full_paths(history_file_path, output_models_path, input_data_folder):
#     def is_base_filename(path):
#         return os.path.basename(path) == path

#     # Use input data folder as the base directory
#     base_dir = input_data_folder

#     # Construct full paths if only base filenames are provided
#     if is_base_filename(history_file_path):
#         history_file_path = os.path.join(base_dir, history_file_path)

#     if is_base_filename(output_models_path):
#         output_models_path = os.path.join(base_dir, output_models_path)

#     return history_file_path, output_models_path

# Function to construct full paths for history and output models
# def construct_full_paths(history_file_path, output_models_path, input_data_folder):
#     def is_base_filename(path):
#         return os.path.basename(path) == path

#     # Use input data folder as the base directory
#     base_dir = input_data_folder

#     # Construct full paths if only base filenames are provided
#     if is_base_filename(history_file_path):
#         history_file_path = os.path.join(base_dir, history_file_path)

#     if is_base_filename(output_models_path):
#         output_models_path = os.path.join(base_dir, output_models_path)

#     # Debugging: Show constructed paths
#     st.write(f"Constructed history file path: {history_file_path}")
#     st.write(f"Constructed output models path: {output_models_path}")

#     return history_file_path, output_models_path


# Function to construct full paths for history and output models
def construct_full_paths(history_file_path, output_models_path, input_data_folder):
    def is_base_filename(path):
        return os.path.basename(path) == path

    # Default save directory is the input data folder, but if it's temporary, use current working directory
    if '/tmp/' in input_data_folder:  # Detecting Streamlit's temporary directory
        base_dir = os.getcwd()  # Save in the current working directory
    else:
        base_dir = input_data_folder  # Use input folder if it's a real directory

    # Construct full paths if only base filenames are provided
    if is_base_filename(history_file_path):
        history_file_path = os.path.join(base_dir, history_file_path)

    if is_base_filename(output_models_path):
        output_models_path = os.path.join(base_dir, output_models_path)
        
    # Debugging: Show constructed paths
    st.write(f"Constructed history file path: {history_file_path}")
    st.write(f"Constructed output models path: {output_models_path}")

    return history_file_path, output_models_path






# Function to save uploaded files to a temporary directory
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    file_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    return file_paths







# Layout with two columns
left_col, right_col = st.columns([2, 3])  # Adjust ratios as needed

# Left Column: File Uploads and CLI Arguments
with left_col:
    # Sidebar for File Upload or Directory Path
    st.header("📂 Upload Data Files")

    # File uploader for joblib and CSV files
    uploaded_files = st.file_uploader(
        "Drag and drop or browse your files (.joblib, .csv)",
        type=["joblib", "csv"],
        accept_multiple_files=True
    )

    # Input for directory path (as an alternative)
    directory_path = st.text_input("Or enter a directory path:")

    # Display uploaded files or directory path
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
        for file in uploaded_files:
            st.write(f"📄 {file.name}")
    elif directory_path:
        if os.path.isdir(directory_path):
            st.success(f"Directory selected: {directory_path}")
        else:
            st.error("Invalid directory path. Please enter a valid path.")
    else:
        st.info("Please upload at least one file or enter a directory path.")


    # Sidebar for CLI Arguments
    st.header("⚙️ Configure Optimization Parameters")

    # CLI Argument Inputs
    model = st.text_input("Model:", value='gpt-4o-mini')
    model_provider = st.text_input("Model Provider (Optional):", value='openai')
    history_file_path = st.text_input("History Model/Metrics File Path:", value='model_history.joblib')
    iterations = st.number_input("Number of Iterations:", min_value=1, max_value=100, value=10)
    extra_info = st.text_area("Extra Info for the LLM:", value='Not available')
    output_models_path = st.text_input("Trained Model Path (Optional):")
    is_regression = st.selectbox("Is Regression?", options=["None", "true", "false"], index=0)
    metrics_source = st.selectbox("Metrics Source:", options=["validation", "test"], index=0)
    error_model = st.text_input("Error Model (Optional):")

# Right Column: Real-time Console Output
with right_col:
    st.header("🖥️ Output")

    # Placeholder for dynamic console output
    output_placeholder = st.empty()

    if st.button("Run Optimization"):
        # Determine data path (file or directory)
        if uploaded_files and len(uploaded_files) == 1:
            saved_files = save_uploaded_files(uploaded_files)
            data_path = saved_files[0]
            input_data_folder = os.path.dirname(data_path)  # This will be a temp folder for uploads
    
        elif directory_path:
            data_path = directory_path
            input_data_folder = directory_path
    
        else:
            st.error("Please upload a file or enter a directory path before running.")
            st.stop()
    
        # Construct full paths for history and output models
        history_file_path, output_models_path = construct_full_paths(
            history_file_path,
            output_models_path,
            input_data_folder
        )
    
        st.write(f"Constructed history file path: {history_file_path}")
        st.write(f"Constructed output models path: {output_models_path}")
    
        # Prepare arguments
        args_dict = {
            'model': model,
            'model_provider': model_provider,
            'history_file_path': history_file_path,
            'iterations': iterations,
            'extra_info': extra_info,
            'output_models_path': output_models_path,
            'is_regression': is_regression,
            'metrics_source': metrics_source,
            'error_model': error_model
        }
    
        # Run CLI and display real-time output in the right column using the placeholder
        run_cli_with_output(data_path, args_dict, output_placeholder)








# # Sidebar for File Upload or Directory Path
# st.header("📂 Upload Data Files")

# # File uploader for joblib and CSV files
# uploaded_files = st.file_uploader(
#     "Drag and drop or browse your files (.joblib, .csv)",
#     type=["joblib", "csv"],
#     accept_multiple_files=True
# )

# # Input for directory path (as an alternative)
# directory_path = st.text_input("Or enter a directory path:")

# # Display uploaded files or directory path
# if uploaded_files:
#     st.success(f"{len(uploaded_files)} file(s) uploaded.")
#     for file in uploaded_files:
#         st.write(f"📄 {file.name}")

# elif directory_path:
#     if os.path.isdir(directory_path):
#         st.success(f"Directory selected: {directory_path}")
#     else:
#         st.error("Invalid directory path. Please enter a valid path.")

# else:
#     st.info("Please upload at least one file or enter a directory path.")





# # Sidebar for CLI Arguments
# st.header("⚙️ Configure Optimization Parameters")

# # Default values for the arguments
# default_model = 'gpt-4o-mini'
# default_history_file_path = 'model_history.joblib'
# default_iterations = 10
# default_extra_info = 'Not available'
# default_metrics_source = 'validation'

# # 1. Model Selection
# model = st.text_input("Model:", value=default_model)

# # 2. Model Provider (Optional)
# model_provider = st.text_input("Model Provider (Optional):", value='openai')

# # 3. History File Path
# history_file_path = st.text_input("History Model/Metrics File Path:", value=default_history_file_path)

# # 4. Number of Iterations
# iterations = st.number_input("Number of Iterations:", min_value=1, max_value=100, value=default_iterations)

# # 5. Extra Info
# extra_info = st.text_area("Extra Info for the LLM:", value=default_extra_info)

# # 6. Output Models Path (Optional)
# output_models_path = st.text_input("Trained Model Path (Optional):")

# # 7. Is Regression
# is_regression = st.selectbox("Is Regression?", options=["None", "true", "false"], index=0)

# # 8. Metrics Source
# metrics_source = st.selectbox("Metrics Source:", options=["validation", "test"], index=0)

# # 9. Error Model (Optional)
# error_model = st.text_input("Error Model (Optional):")


# # Function to construct full paths for history and output models
# def construct_full_paths(history_file_path, output_models_path, input_data_folder):
#     def is_base_filename(path):
#         return os.path.basename(path) == path

#     # Use input data folder as the base directory
#     base_dir = input_data_folder

#     # Construct full paths if only base filenames are provided
#     if is_base_filename(history_file_path):
#         history_file_path = os.path.join(base_dir, history_file_path)

#     if is_base_filename(output_models_path):
#         output_models_path = os.path.join(base_dir, output_models_path)

#     return history_file_path, output_models_path


# # Function to run CLI with arguments and display output
# def run_cli_with_output(data_path, args_dict):
#     # Build the CLI command
#     command = ['hs_optimize', '--data', data_path]

#     for key, value in args_dict.items():
#         if value is not None and value != "None":  # Skip None values
#             cli_key = f'--{key.replace("_", "-")}'
#             command.append(cli_key)
#             command.append(str(value))

#     # Display the command for debugging (optional)
#     st.code(' '.join(command), language='bash')

#     # Run the command and capture real-time output
#     placeholder = st.empty()
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     output = ""
#     for line in iter(process.stdout.readline, ''):
#         output += line
#         placeholder.text(output)  # Update the output panel in real-time
#         time.sleep(0.1)

#     process.stdout.close()
#     process.wait()

#     # Final output after process finishes
#     placeholder.text(output)
#     st.success("Optimization completed!")

# import tempfile
# import shutil

# # Function to save uploaded files to a temporary directory
# def save_uploaded_files(uploaded_files):
#     temp_dir = tempfile.mkdtemp()  # Create a temporary directory
#     file_paths = []

#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(temp_dir, uploaded_file.name)
#         with open(file_path, 'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         file_paths.append(file_path)

#     return file_paths

# # Button to run the CLI
# if st.button("Run Optimization"):
#     # Determine data path (file or directory)
#     if uploaded_files and len(uploaded_files) == 1:
#         saved_files = save_uploaded_files(uploaded_files)  # Save uploaded file temporarily
#         data_path = saved_files[0]  # Use full path of uploaded file
#         input_data_folder = os.path.dirname(data_path)  # Get folder of uploaded file
#     elif directory_path:
#         data_path = directory_path  # Use provided directory path
#         input_data_folder = directory_path  # Folder is the directory itself
#     else:
#         st.error("Please upload a file or enter a directory path before running.")
#         st.stop()

#     # Construct full paths for history and output models in the input data folder
#     history_file_path, output_models_path = construct_full_paths(
#         history_file_path,
#         output_models_path,
#         input_data_folder  # Use the input data folder as base directory
#     )

#     # Prepare arguments dictionary
#     args_dict = {
#         'model': model,
#         'model_provider': model_provider,
#         'history_file_path': history_file_path,
#         'iterations': iterations,
#         'extra_info': extra_info,
#         'output_models_path': output_models_path,
#         'is_regression': is_regression,
#         'metrics_source': metrics_source,
#         'error_model': error_model
#     }

#     # Run CLI and display output
#     run_cli_with_output(data_path, args_dict)



