# Interpretable Graph-Based Stock Trading Decision System

## Instruction for Running the Experiments

### Dataset
First download the dataset from [S&P 500 Stocks](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks). Place the `sp500_companies.csv`, `sp500_index.csv` and `sp500_stocks.csv` in the ./data-provider/Datasets/sp500 directory. 

The structure should look like this:
```
project-root/
├── data-provider/
│   ├── datasets.py
│   ├── Datasets/
│       ├── sp500/
│           ├── sp500_companies.csv
│           ├── sp500_index.csv
│           └── sp500_stocks.csv
├── modelling/
│   └── ...
├── util/
│   └── ...
├── train.ipynb
├── dataset_analysis.ipynb
├── README.md
└── ...
```
### Upload to Google Drive

Then upload these files to Google Drive in the `/content/drive/MyDrive/Colab Notebooks/winter24-project/TrustProAuto-team-main/` directory. If you prefer a different location, update the path accordingly in the first cell of `train.ipynb`.

The experiments in this project are designed to be run on Google Colab. All the required dependencies should be satisfied or installed by running the notebook. 

The main files are:

- `train.ipynb`: Main script to run the experiment, including constructing graph, analyzing sparsity, training graph neural networks, and explaining the output. 
- `dataset_analysis.ipynb`: Script to analyze the dataset, including trend plot, stationarity test, stationarization using log differencing, correlation matrix, mutual information and time lagged analysis.

