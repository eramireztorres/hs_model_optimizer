from setuptools import setup, find_packages

setup(
    name='hs-model-optimizer',
    version='1.0.0',
    description='A project for hot-swapping model optimization using LLM suggestions.',
    author='Erick Eduardo Ramirez Torres',
    author_email='erickeduardoramireztorres@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn',
        'xgboost',
        'joblib',
        'logging',
        'openai'
    ],
    entry_points={
        'console_scripts': [
            'hs_optimize=cli:main'
        ],
    },
)

