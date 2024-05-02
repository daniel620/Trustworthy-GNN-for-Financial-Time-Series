# Interpretable Graph-Based Stock Trading Decision System

## Instruction for Running the Experiments

### Dataset
First download the dataset from [S&P 500 Stocks](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks). Place the `sp500_companies.csv`, `sp500_index.csv` and `sp500_stocks.csv` in the Datasets/sp500 directory. 

All the code files are located in the root directory of the project. The structure should look like this:
```
project-root/
├── Datasets/
│   ├── sp500
│       ├── sp500_companies.csv
│       └── sp500_index.csv
│       └── sp500_stocks.csv
├── train.ipynb
├── dataset_analysis.ipynb
├── README.md
└── ...
```

The main files are:

- `train.ipynb`: Main script to run the experiment, including constructing graph, analyzing sparsity, training graph neural networks, and explaining the output. 
- `dataset_analysis.ipynb`: Script to analyze the dataset, including trend plot, stationarity test, stationarization using log differencing, correlation matrix, mutual information and time lagged analysis. 

### Deep Learning Dependencies
- torch: 2.0.1

- torch-geometric: 2.3.1
