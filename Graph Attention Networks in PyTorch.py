# I'm importing necessary libraries and modules for graph operations, neural networks, and visualization.
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

# Here are the basic components of our graph: nodes, edges, and labels for each node.
NODES = [0, 1, 2]
EDGES = [(0, 1), (0, 2)]
LABELS = [0, 1, 2]

class Graph:
    def __init__(self, nodes, edges):
        # Upon initialization, I create a graph with the given nodes and edges.
        self.graph = self.create_graph(nodes, edges)

    def create_graph(self, nodes, edges):
        # This method creates and returns a graph object using NetworkX.
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    def get_adjacency_matrix(self):
        # I'm obtaining the adjacency matrix of the graph.
        return nx.adjacency_matrix(self.graph).todense()

    def get_labels(self, labels):
        # Here, I convert labels into a tensor for neural network processing.
        return torch.tensor(labels, dtype=torch.long)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha):
        # When initializing a GAT layer, I set up the weights and necessary parameters.
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att_weights = nn.Parameter(torch.Tensor(out_dim, 1))
        nn.init.xavier_uniform_(self.att_weights.data, gain=1.414)
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, X, A):
        # During the forward pass, I compute the attention mechanism and return the updated node features.
        h = self.W(X)
        a = torch.matmul(h, self.att_weights)
        zero_vec = -9e15 * torch.ones_like(a)
        attention = torch.where(A > 0, a, zero_vec)
        attention = F.softmax(F.leaky_relu(attention, negative_slope=self.alpha), dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return h_prime

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, alpha):
        # The GAT model is setup here with a sequence of layers and a final classifier.
        super(GAT, self).__init__()
        self.layers = nn.ModuleList([
            GATLayer(in_dim, hidden_dim, dropout, alpha),
            GATLayer(hidden_dim, out_dim, dropout, alpha)
        ])
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, X, A):
        # I pass the node features and adjacency matrix through each layer in the forward pass.
        H = X
        for layer in self.layers:
            H = layer(H, A)
        out = self.fc(H)
        return F.log_softmax(out, dim=1)

class Trainer:
    def __init__(self, model, optimizer, criterion):
        # I'm initializing the trainer with a model, optimizer, and loss function.
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.losses = []

    def train(self, X, A, labels, epochs):
        # Here, I train the model over a specified number of epochs and record the loss.
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            logits = self.model(X, A)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def plot_losses(self):
        # I'm plotting the training loss over epochs using Plotly.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(self.losses)+1)), y=self.losses, mode='lines'))
        fig.update_layout(title="GAT Training Loss", xaxis_title="Epoch", yaxis_title="Cross-Entropy Loss")
        fig.show()

class GATApp:
    def __init__(self, nodes, edges, labels, in_dim=1, hidden_dim=8, out_dim=3, heads=2, dropout=0.6, lr=0.01):
        # In the app's initialization, I setup the graph, model, and optimization parameters.
        self.graph = Graph(nodes, edges)
        self.labels = self.graph.get_labels(labels)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.model = self._initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _initialize_model(self):
        # This method initializes the GAT model with specified parameters.
        return GAT(self.in_dim, self.hidden_dim, self.out_dim, self.dropout, self.heads)

    def run(self):
        # I'm running the training process and visualizing node representations and losses.
        trainer = Trainer(self.model, self.optimizer, nn.CrossEntropyLoss())
        adjacency_matrix = torch.from_numpy(self.graph.get_adjacency_matrix()).float()
        trainer.train(torch.ones((len(self.graph.graph.nodes), self.in_dim)), adjacency_matrix, self.labels, epochs=100)

        trainer.plot_losses()

        with torch.no_grad():
            node_reps = self.model(torch.ones((len(self.graph.graph.nodes), self.in_dim)), adjacency_matrix).numpy()

        fig = go.Figure()
        for i in range(len(self.graph.graph.nodes)):
            fig.add_trace(go.Scatter(x=[node_reps[i, 0]], y=[node_reps[i, 1]], mode='markers',
                                     marker=dict(size=10), text=f"Class {self.labels[i].item()}"))
        fig.update_layout(title="GAT Node Representations", xaxis_title="Dimension 1", yaxis_title="Dimension 2")
        fig.show()

app = GATApp(NODES, EDGES, LABELS)
app.run()