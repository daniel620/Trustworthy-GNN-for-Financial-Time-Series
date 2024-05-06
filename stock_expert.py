import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from data_provider.datasets import SP500Dataset, TSDataset
from util.my_visualize_graph import visualize_graph


class Agent:
    def __init__(self, weight_path):
        # Load model and graph constructor
        self.model = torch.load(os.path.join(weight_path, 'model.pth')).to('cpu')
        self.window_size = self.model.window_size
        self.gc = pickle.load(open(os.path.join(weight_path, 'gc.pkl'), 'rb'))

        # Load data
        self.data_numpy = SP500Dataset().get_data_numpy(log_return=False)
        self.dataset = TSDataset(self.data_numpy, self.window_size, horizon=22, stride=1)

        # Load industry data
        self.stock_dataset = SP500Dataset()
        self.industries = self.stock_dataset.get_industry().loc[self.stock_dataset.get_names()]
        self.data_df = self.stock_dataset.stocks_return_df

        print('Model information:')
        print(self.model)

    def get_adjacency_matrix(self, tho_adj=0.1, show_networkx_plot=False):
        # Generate adjacency matrix from graph constructor
        adj = self.gc.get_adjacency_matrix(tho_adj)
        if show_networkx_plot:
            G = nx.from_numpy_matrix(adj)
            pos = nx.spring_layout(G)
            plt.figure(figsize=(4, 4))
            nx.draw(G, pos=pos, with_labels=False, node_size=1, width=0.2)
            plt.show()
        return adj

    def draw_heat_map(self, corr, sign=False, show_plot=True):
        if sign:
            corr = np.sign(corr)  # Apply sign to highlight positive or negative correlations

        # Sort indices by industry sector
        sorted_industries = self.industries.sort_values('Sector')
        sorted_indices = [self.industries.index.get_loc(stock) for stock in sorted_industries.index]
        sorted_corr = corr[:, sorted_indices][sorted_indices]

        # Plotting
        plt.imshow(sorted_corr, aspect='auto', cmap='coolwarm')
        plt.colorbar()

        # Sector boundaries
        sector_counts = sorted_industries['Sector'].value_counts().sort_index()
        sector_positions = sector_counts.cumsum()
        sector_labels = sector_counts.index
        tick_positions = sector_positions - sector_counts / 2

        plt.yticks(tick_positions, sector_labels, rotation='horizontal')

        for pos in sector_positions[:-1]:
            plt.axhline(y=pos - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.3)
            plt.axvline(x=pos - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.3)

        if show_plot:
            plt.show()

    def plot_network_by_industry(self, adj, layout='spring', node_size=50, edge_width=0.5, show_plot=True):
        # Sort the industries and correspondingly sort the adjacency matrix
        sorted_industries = self.industries.sort_values('Sector')
        sorted_indices = [self.industries.index.get_loc(stock) for stock in sorted_industries.index]
        sorted_adj = adj[np.ix_(sorted_indices, sorted_indices)]

        # Generate network graph from sorted adjacency matrix
        G = nx.from_numpy_matrix(sorted_adj)

        # Creating a color map from sectors to colors
        color_map = {sector: plt.cm.tab20(i % 20) for i, sector in enumerate(sorted(sorted_industries['Sector'].unique()))}
        node_colors = [color_map[sorted_industries.iloc[node]['Sector']] for node in G.nodes()]

        # Layout configuration
        if layout == 'spring':
            pos = nx.spring_layout(G)
        else:
            pos = nx.circular_layout(G)

        plt.figure(figsize=(12, 12))  # Adjusted for better visibility
        nx.draw(G, pos=pos, node_size=node_size, width=edge_width, node_color=node_colors, with_labels=False, alpha=0.8)

        # Legend for sectors
        if show_plot:
            plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=sector, markerfacecolor=color, markersize=10)
                                for sector, color in color_map.items()])
            plt.show()

    def generate_explanations(self, node_index, date='2023-10-12'):
        date_index = self.stock_dataset.stocks_return_df.index.get_loc(date)
        x, _ = self.dataset[date_index]
        if x.shape[1] != self.window_size:
            print(x.shape, self.window_size)
            raise ValueError('Invalid date or window size.')

        adj = self.get_adjacency_matrix(tho_adj=0.1, show_networkx_plot=False)
        edge_index = torch.nonzero(torch.tensor(adj), as_tuple=False).t().contiguous()

        # Initialize the explainer with the GNN model
        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            )
        )

        # Generate explanation for the specified node
        explanation = self.explainer(x, edge_index, index=node_index)

        # Calculate 2-hop influences
        node_labels = self.stock_dataset.get_names()
        target_node_name = node_labels[node_index]
        # calculate historical volatility
        price = x[node_index]
        history_return_rate = (price[1:] - price[:-1]) / price[:-1]
        volatility = history_return_rate.std()
        pred_price = explanation.prediction[node_index][0]
        pred_return_rate = (explanation.prediction[node_index][0] - price[-1]) / price[-1]

        # draw price trend, pred value different color
        horizon = 7
        plt.plot(price, label='Price')
        plt.axvline(x=self.window_size - 1, color='r', linestyle='--', label='Prediction Start')
        plt.axvline(x=self.window_size - 1 + horizon, color='g', linestyle='--', label='Prediction End')
        plt.axhline(y=explanation.prediction[node_index][0], color='b', linestyle='--', label='Predicted Price')
        # draw volatility for pred
        plt.fill_between([self.window_size - 1, self.window_size - 1 + horizon], pred_price * (1 - volatility),
                         pred_price * (1 + volatility), color='gray', alpha=0.2, label='Volatility')
        plt.title(f"Price Trend for {target_node_name}")
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        print(f"Predicted average value for {target_node_name} in the next 7 days is estimated at ${explanation.prediction[node_index][0]:.2f}.")
        print(f"Predicted return rate is estimated at {pred_return_rate:.4f} on average, with a volatility of {volatility:.4f}")

        # Dictionary to hold direct and indirect influences
        contributing_nodes = {}

        # First, handle direct contributions
        target_edges = torch.where(explanation.edge_index[1] == node_index)[0]
        for edge in target_edges:
            node_id = explanation.edge_index[0, edge].item()
            weight = explanation.edge_mask[edge].item()
            contributing_nodes[node_id] = contributing_nodes.get(node_id, 0) + weight

        # Second, handle 2-hop contributions
        for node_id in list(contributing_nodes.keys()):
            intermediate_edges = torch.where(explanation.edge_index[1] == node_id)[0]
            for edge in intermediate_edges:
                source_node_id = explanation.edge_index[0, edge].item()
                weight = explanation.edge_mask[edge].item()
                total_contribution = contributing_nodes[node_id] * weight
                contributing_nodes[source_node_id] = contributing_nodes.get(source_node_id, 0) + total_contribution

        sorted_contributing_nodes = sorted(contributing_nodes.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"Here’s how the nodes influence {target_node_name}'s forecast:")
        for node_id, importance in sorted_contributing_nodes:
            node_name = node_labels[node_id]
            print(f" - {node_name} has an importance score of {importance:.4f}.")

        print('To better manage risk, you are sugested to consider the following stocks:')
        # find 3 stocks with lowest correlation in self.gc.get_precision_matrix(), which may be negatively correlated
        corr = self.gc.get_precision_matrix()
        sorted_corr = corr[node_index].argsort()
        for idx in sorted_corr[:3]:
            print(f" - {node_labels[idx]} with estimated correlation score of {corr[node_index][idx]:.4f}.")

        return explanation

    def visualize_explanations(self, explanation, node_labels, node_index, top_n=10):
        # Visualizing the influence network graphically
        g = visualize_graph(edge_index=explanation.edge_index, edge_weight=explanation.edge_mask, backend='graphviz',
                            node_labels=node_labels)

        from IPython.display import display
        display(g)

        # Creating a color bar to illustrate weight significance
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cmap = plt.cm.get_cmap('viridis')
        norm = plt.Normalize(vmin=0, vmax=1)
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
        cb.set_label('Weight Importance')
        plt.show()


if __name__ == "__main__":
    agent = Agent('weights/0')
    node_idx = 0
    explanation = agent.generate_explanations(node_index=node_idx)
    agent.visualize_explanations(explanation, node_index=node_idx, node_labels=agent.stock_dataset.get_names())
