from setuptools import setup, find_packages

setup(
    name='hs-model-optimizer',
    version='1.0.0',
    description='A project for hot-swapping model optimization using LLM suggestions.',
    author='Erick Eduardo Ramirez Torres',
    author_email='erickeduardoramireztorres@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'scikit-learn',
        'xgboost',
        'joblib',
        'openai'
    ],
    entry_points={
        'console_scripts': [
            'hs_optimize=src.cli:select_model_cli'  
        ],
    },
)

