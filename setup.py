from setuptools import setup, find_packages

setup(
    name="stocks_miner",
    version="0.1.0",
    author="Soumyadip Das, Rajdeep Chatterjee",
    author_email="dassoumyadip204@gmail.com, cse.rajdeep@gmail.com",
    description="A Python package for analyzing and detecting stock market crashes using TDA and ML.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Soumyadippallab/Stocks_Miner",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas==2.3.2",
        "numpy==1.26.4",
        "yfinance==0.2.66",
        "scikit-learn==1.6.1",
        "scipy==1.13.1",
        "matplotlib==3.9.4",
        "seaborn==0.13.2",
        "yahooquery==2.4.1",
        "tqdm==4.67.1",
        "requests==2.32.4",
        "ripser==0.6.12",
        "persim==0.3.8"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
