import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="absehrd",
    version="0.0.1",
    author="Haley Hunter-Zinck",
    author_email="haleyhunterzinck@berkeley.edu",
    description="Automated Brewing of Synthetic Electronic Health Record Data",
    long_description="Consolidated pipeline for producing and validating generators for synthetic eletronic health record data.",
    long_description_content_type="text/markdown",
    url="https://github.com/Innovate-For-Health/absehrd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
