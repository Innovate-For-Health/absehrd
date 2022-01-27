from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Automated Brewing of Synthetic Electronic Health Record Data'
LONG_DESCRIPTION = 'Trains a generator for synthetic electronic health record data, generates synthetic data from the model, and validates the synthetic dataset.'

setup(name="absehrd", 
        version="0.0.1",
        author="Haley Hunter-Zinck",
        author_email="haley.hunterzinck@sagebase.org",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["pyarrow==0.16.0", "matplotlib==3.3.1", "numpy==1.19.1", "scipy==1.5.2", "scikit-learn==0.23.2", "torch==1.4.0", "tqdm==4.31.1", "pandas==1.1.3", "pytest==6.2.4"],
        python_requires='>=3.7, <3.9',
        entry_points={'console_scripts': ['absehrd = '
                    'absehrd.__main__:main']},
        keywords=['python', 'synthetic data', 'electronic health record']
)
