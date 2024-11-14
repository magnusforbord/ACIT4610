**Portfolio Optimization Project**

This project involves analyzing and optimizing a portfolio of 20 stocks using Evolutionary Programming (EP) and Evolution Strategy (ES) algorithms. The goal is to determine the optimal portfolio that maximizes returns while minimizing risk, using historical stock data from 2018 to 2022.

**Project Structure**

* data/: Contains the raw and processed stock data used for analysis.

* raw/: Stores raw CSV files for each stock.

* processed/: Stores processed data, including monthly returns and covariance matrices.

* figures/: Directory for saving figures generated during analysis.

* results/: Directory for storing results from the EP and ES algorithms.

**scripts:**

* analyze_results.py: Script to analyze the results from different algorithms.

* data_preprocessing.py: Preprocesses raw stock data and saves it in a usable format.

* ep_basic.py, ep_advanced.py: Implementation of Evolutionary Programming (EP) algorithms.

* es_basic.py, es_advanced.py: Implementation of Evolution Strategy (ES) algorithms.

* es_mu_comma_lambda.py, es_mu_plus_lambda.py: Specific ES algorithms with (μ, λ) and (μ + λ) strategies.

* README.md: Project documentation.

* requirements.txt: List of required Python packages.

* utils.py: Contains utility functions for data processing and analysis.

**Installation:**

Clone this repository.

Install dependencies using:
pip install -r requirements.txt

**Data Collection and Preprocessing**

The process of collecting historical stock data and preparing it for optimization is automated through the data_preprocessing.py script.

Steps:

1. Data Collection: The script downloads daily stock data for the specified tickers using the yfinance library and saves each stock's data as a CSV file in the data/raw directory.

2. Preprocessing:

Ensures required columns are present and verifies data integrity.

Converts the 'Date' column to a datetime format and sets it as the index.

Extracts 'Close' prices, combines them into a single DataFrame, and resamples to monthly prices.

Calculates monthly returns and the covariance matrix of these returns.

Saves the processed data to the data/processed directory.

To run the data preprocessing script, execute:
python data_preprocessing.py

**Approach**

Evolutionary Programming (EP)

* Basic EP (ep_basic.py): Implements a basic form of evolutionary programming for portfolio optimization.
* Advanced EP (ep_advanced.py): Implements an advanced version with additional features, such as selection and mutation strategies.

Evolution Strategy (ES)

* Basic ES (es_basic.py): Implements a basic ES algorithm focusing on portfolio optimization.
* Advanced ES (es_advanced.py): Includes additional features to enhance the optimization process.

* (μ, λ) and (μ + λ) Strategies: The scripts es_mu_comma_lambda.py and es_mu_plus_lambda.py implement these strategies to explore different selection and reproduction mechanisms.

**Running the Optimization**
Each algorithm can be run individually to perform optimization. For example, to run the basic EP algorithm:

* python ep_basic.py

Similarly, other algorithms can be run using their respective script files. The results are saved in the results directory.


**Results Analysis**

After running the optimization algorithms, use analyze_results.py to generate summaries and visualizations of the results:

* python analyze_results.py


**Conclusion**

This project provides a foundation for understanding and implementing evolutionary algorithms for portfolio optimization. It combines data processing with advanced algorithmic techniques to optimize stock portfolios.
