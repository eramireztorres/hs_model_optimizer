from setuptools import setup, find_packages

setup(
    name='hs-model-optimizer',
    version='1.0.0',
    description='A project for hot-swapping model optimization using LLM suggestions.',
    author='Erick Eduardo Ramirez Torres',
    author_email='erickeduardoramireztorres@gmail.com',
    packages=find_packages(),
    include_package_data=True,  # Ensure package data is included
    package_data={
        '': ['prompts/*.txt'],  # Include all .txt files in prompts folder
    },
    install_requires=[
        'scikit-learn==1.5.2',  # Ensure compatibility with your codebase
        'xgboost==2.1.1',
        'joblib==1.4.2',
        'openai==1.45.0',
        'lightgbm==4.5.0',
        'pytz==2024.2',
        'requests==2.32.3',
        'pandas==2.2.3',
        'httpx==0.27.2',
        'catboost==1.2.7',
        'streamlit==1.41.1'
        
    ],
    entry_points={
        'console_scripts': [
            'hs_optimize=src.cli:select_model_cli'  
        ],
    },
)

