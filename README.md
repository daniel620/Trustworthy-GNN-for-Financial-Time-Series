# Interpretable Graph-Based Stock Trading Decision System

## Features

1. **Probabilistic Graph Construction**: Utilizes the **Graphical Lasso (GLasso)** algorithm to probabilistically infer graphs from **multivariate time series data**. This method effectively captures the underlying structure and conditional dependencies between different stocks.

2. **Robust Data Augmentation**: Incorporates **Ornstein-Uhlenbeck (OU) noise** as both **adversarial and data augmentation** technique. This approach enhances model robustness by simulating realistic financial scenarios and improving the network's resilience against noise.

3. **Modular Time Series Embedders**: Modular architecture that supports various embedders for time series feature extraction
   - 1D convolutional neural networks
   - LSTM
   - ... 

4. **Modular GNN Architecture**: Features a modular design for the Graph Neural Network (GNN) component, allowing for interchangeable message passing layers
   - GCN (Graph Convolutional Network)
   - GIN (Graph Isomorphism Network)
   - ...

5. **Explainable Predictions with GNNExplainer**: Leverages the **GNNExplainer** module to provide detailed explanations of prediction results, highlighting influential factors and their contributions. This transparency aids in understanding and trusting model decisions.

6. **Integrated Gradients Explanation**: Employs **Integrated Gradients (IG)** to elucidate the influence of information at each timestep in a time series on the outcome. By leveraging IG, importance scores are attributed to individual timesteps, providing detailed insight into how each specific input feature contributes to the model’s prediction. 

7. **Automated Decision Process**: Implements a fully **automated decision-making** framework that integrates data preprocessing, model training, and inference to deliver actionable trading insights. This automation streamlines the trading strategy, reducing the need for manual intervention.

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
### Running on Google Colab

Upload these files to Google Drive in the `/content/drive/MyDrive/Colab Notebooks/winter24-project/TrustProAuto-team-main/` directory. If you prefer a different location, update the path accordingly in the first cell of the notebook.

The experiments in this project are designed to be run on Google Colab. All the required dependencies should be satisfied or installed by running the notebook. 

The main files are:

- `train.ipynb`: Main script to run the experiment, including constructing graph, analyzing sparsity, training graph neural networks, and explaining the output. 
- `dataset_analysis.ipynb`: Script to analyze the dataset, including trend plot, stationarity test, stationarization using log differencing, correlation matrix, mutual information and time lagged analysis.

### Running Locally

#### Dependencies

Please install the following Python packages:

```bash
pip install graphviz==0.20.1 \
             ipython==8.12.3 \
             matplotlib==3.7.2 \
             networkx==2.8.8 \
             numpy==1.24.4 \
             pandas==1.3.5 \
             scikit_learn==1.3.0
```
To install `torch` and `torch_geometric`, which have specific platform-dependent installation procedures, please refer to the official guides:
- [`torch`](https://pytorch.org/get-started/locally/) -  `torch==2.0.1`
- [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) - `torch_geometric==2.3.1`

