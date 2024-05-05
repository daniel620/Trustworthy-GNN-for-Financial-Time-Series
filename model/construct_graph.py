
from sklearn.covariance import GraphicalLasso,GraphicalLassoCV
import numpy as np
import matplotlib.pyplot as plt

class GraphConstructor:
    def __init__(self,data:np.array, names:list=[], alpha=None, max_iter=100):
        
        self.names = names.copy()
        self.data = data.copy()

        print("Normalizing the data's variance...")
        self.data /= self.data.std(axis=0)

        self.T, self.n = data.shape
        print(f'{self.n} nodes and {self.T} time samples')

        if alpha is None:
            self.model = GraphicalLassoCV(max_iter=max_iter)
        else:
            self.model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        
        print("Fitting the model...")
        self.model.fit(self.data)
    
    def get_precision_matrix(self):
        return self.model.precision_

    def get_alpha(self):
        if isinstance(self.model, GraphicalLassoCV):
            return self.model.alpha_
        elif isinstance(self.model, GraphicalLasso):
            return self.model.alpha
        else:
            raise ValueError("Model not recognized")
    
    def get_adjacency_matrix(self, threshold=0.3):
        precision = self.get_precision_matrix()
        adjacency_matrix = np.abs(precision) >= threshold
        # remove self loops
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix.astype(int)
    
    def draw_graph(self, threshold=0.3):
        import networkx as nx
        import matplotlib.pyplot as plt
        adjacency_matrix = self.get_adjacency_matrix(threshold)
        G = nx.from_numpy_matrix(adjacency_matrix)
        if len(self.names) == self.n:
            G = nx.relabel_nodes(G, dict(enumerate(self.names)))
        nx.draw(G, with_labels=True)
        plt.show()


def draw_heat_map(corr, sign=False):
    if sign:
        corr = np.sign(corr)
    plt.imshow(corr)
    plt.colorbar()
    plt.show()