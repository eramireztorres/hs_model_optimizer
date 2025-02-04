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


def construct_full_paths(history_file_path, output_models_path, input_data_folder, directory_path=None, uploaded_files=None):
    def is_base_filename(path):
        return os.path.basename(path) == path

    # If directory_path is provided, use it
    if directory_path:
        base_dir = directory_path
    # If the file was uploaded, use the 'outputs' folder
    elif uploaded_files:
        base_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(base_dir, exist_ok=True)
    # Otherwise, use the input data folder (for browsed files)
    else:
        base_dir = input_data_folder

    # Construct full paths if only base filenames are provided
    if is_base_filename(history_file_path):
        history_file_path = os.path.join(base_dir, history_file_path)

    # Return None if output_models_path is None or empty
    if not output_models_path or output_models_path.strip() == "":
        output_models_path = None
    elif is_base_filename(output_models_path):
        output_models_path = os.path.join(base_dir, output_models_path)

    return history_file_path, output_models_path


# def construct_full_paths(history_file_path, output_models_path, input_data_folder, directory_path=None):
#     def is_base_filename(path):
#         return os.path.basename(path) == path

#     # If directory_path is provided, use it; otherwise, use input_data_folder
#     base_dir = directory_path if directory_path else input_data_folder

#     if not base_dir:
#         base_dir = os.path.join(os.getcwd(), "outputs")  # Default outputs directory
#         os.makedirs(base_dir, exist_ok=True)

#     # Construct full paths if only base filenames are provided
#     if is_base_filename(history_file_path):
#         history_file_path = os.path.join(base_dir, history_file_path)

#     # Return None if output_models_path is None or empty
#     if not output_models_path or output_models_path.strip() == "":
#         output_models_path = None
#     elif is_base_filename(output_models_path):
#         output_models_path = os.path.join(base_dir, output_models_path)

#     return history_file_path, output_models_path


# Function to save uploaded files to a temporary directory
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
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
    st.header("📂 Upload Data Files")

    uploaded_files = st.file_uploader(
        "Drag and drop or browse your file (.joblib, .csv). If not provided, multiple csv files will be read from directory path.",
        type=["joblib", "csv"],
        accept_multiple_files=False
    )

    
    directory_path = st.text_input(
    "Enter a directory path (outputs will be saved here if provided, otherwise in 'outputs/' folder):")


    if uploaded_files:
        st.success("1 file uploaded.")  # Always 1 file if uploaded_files is not None
        st.write(f"📄 {uploaded_files.name}")
    elif directory_path:
        if os.path.isdir(directory_path):
            st.success(f"Directory selected: {directory_path}")
        else:
            st.error("Invalid directory path. Please enter a valid path.")
    else:
        st.info("Please upload at least one file or enter a directory path.")


    st.header("⚙️ Configure Optimization Parameters")

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

    output_placeholder = st.empty()

    if st.button("Run Optimization"):
        if uploaded_files:
            saved_files = save_uploaded_files([uploaded_files])  # Wrap in list to reuse the same function
            data_path = saved_files[0]
            input_data_folder = os.path.dirname(data_path)
        elif directory_path:
            data_path = directory_path
            input_data_folder = directory_path
        else:
            st.error("Please upload a file or enter a directory path before running.")
            st.stop()


        # history_file_path, output_models_path = construct_full_paths(
        #     history_file_path,
        #     output_models_path,
        #     input_data_folder,
        #     directory_path
        # )
        
        history_file_path, output_models_path = construct_full_paths(
            history_file_path,
            output_models_path,
            input_data_folder,
            directory_path,
            uploaded_files  # Pass uploaded_files to detect if the file is from Streamlit's temp dir
        )


        st.write(f"Output file path: {history_file_path}")

        # Prepare arguments
        args_dict = {
            'model': model,
            'model_provider': model_provider,
            'history_file_path': history_file_path,
            'iterations': iterations,
            'extra_info': extra_info,
            'metrics_source': metrics_source,
            'error_model': error_model
        }
        
        if is_regression != 'None':
            args_dict['is_regression'] = is_regression.lower()
        
        # Only add 'output_models_path' if it's provided
        if output_models_path:
            args_dict['output_models_path'] = output_models_path


        run_cli_with_output(data_path, args_dict, output_placeholder)
        st.success(f"Optimization completed! Outputs saved in: {os.path.dirname(history_file_path)}")



