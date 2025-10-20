from setuptools import setup, find_packages

setup(
    name="stocks_miner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "yfinance",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "yahooquery",
        "tqdm",
        "requests",
        "ripser",
        "persim"
    ],
    entry_points={
        "console_scripts": [
            "stocks-miner-tda = stocks_miner.cli:tda_command",
            "stocks-miner-nse = stocks_miner.cli:nse_command",
            "stocks-miner-random = stocks_miner.cli:random_command",
            "stocks-miner-indices = stocks_miner.cli:indices_command"
        ]
    },
    author="Soumyadip Das",
    description="A Python package for stock market analysis and crash detection using TDA.",
    license="MIT",
)
