import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import neps
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from neps.utils.common import load_checkpoint, save_checkpoint

from neps_global_utils import process_trajectory, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pipeline_space(searcher) -> dict:  # maybe limiting for ifbo
    """define search space for neps"""
    pipeline_space = dict(
        learning_rate=neps.FloatParameter(
            lower=1e-9,
            upper=10,
            log=True,
        ),
        beta1=neps.FloatParameter(
            lower=1e-4,
            upper=1,
            log=True,
        ),
        beta2=neps.FloatParameter(
            lower=1e-3,
            upper=1,
            log=True,
        ),
        epsilon=neps.FloatParameter(
            lower=1e-12,
            upper=1000,
            log=True,
        )
    )
    uses_fidelity = ("ifbo", "hyperband", "asha", "ifbo_taskset_4p", "ifbo_taskset_4p_extended")
    if searcher in uses_fidelity:
        pipeline_space["epoch"] = neps.IntegerParameter(
            lower=1,
            upper=50,
            is_fidelity=True,
        )
    return pipeline_space


class GraphConv(nn.Module):
    """
    Graph Convolutional Layer described in "Semi-Supervised Classification with Graph Convolutional Networks".

    Given an input feature representation for each node in a graph, the Graph Convolutional Layer aims to aggregate
    information from the node's neighborhood to update its own representation. This is achieved by applying a graph
    convolutional operation that combines the features of a node with the features of its neighboring nodes.

    Mathematically, the Graph Convolutional Layer can be described as follows:

        H' = f(D^(-1/2) * A * D^(-1/2) * H * W)

    where:
        H: Input feature matrix with shape (N, F_in), where N is the number of nodes and F_in is the number of
            input features per node.
        A: Adjacency matrix of the graph with shape (N, N), representing the relationships between nodes.
        W: Learnable weight matrix with shape (F_in, F_out), where F_out is the number of output features per node.
        D: The degree matrix.
    """

    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConv, self).__init__()

        # Initialize the weight matrix W (in this case called `kernel`)
        self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_normal_(
            self.kernel
        )  # Initialize the weights using Xavier initialization

        # Initialize the bias (if use_bias is True)
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)  # Initialize the bias to zeros

    def forward(self, input_tensor, adj_mat):
        """
        Performs a graph convolution operation.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Normalized adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """

        support = torch.mm(
            input_tensor, self.kernel
        )  # Matrix multiplication between input and weight matrix
        output = torch.spmm(
            adj_mat, support
        )  # Sparse matrix multiplication between adjacency matrix and support
        # Add the bias (if bias is not None)
        if self.bias is not None:
            output = output + self.bias

        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) as described in the paper `"Semi-Supervised Classification with Graph
    Convolutional Networks" <https://arxiv.org/pdf/1609.02907.pdf>`.

    The Graph Convolutional Network is a deep learning architecture designed for semi-supervised node
    classification tasks on graph-structured data. It leverages the graph structure to learn node representations
    by propagating information through the graph using graph convolutional layers.

    The original implementation consists of two stacked graph convolutional layers. The ReLU activation function is
    applied to the hidden representations, and the Softmax activation function is applied to the output representations.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True, dropout_p=0.1):
        super(GCN, self).__init__()

        # Define the Graph Convolution layers
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        """
        Performs forward pass of the Graph Convolutional Network (GCN).

        Args:
            input_tensor (torch.Tensor): Input node feature matrix with shape (N, input_dim), where N is the number of nodes
                and input_dim is the number of input features per node.
            adj_mat (torch.Tensor): Normalized adjacency matrix of the graph with shape (N, N), representing the relationships between
                nodes.

        Returns:
            torch.Tensor: Output tensor with shape (N, output_dim), representing the predicted class probabilities for each node.
        """

        # Perform the first graph convolutional layer
        x = self.gc1(input_tensor, adj_mat)
        x = F.relu(x)  # Apply ReLU activation function
        x = self.dropout(x)  # Apply dropout regularization

        # Perform the second graph convolutional layer
        x = self.gc2(x, adj_mat)

        # Apply log-softmax activation function for classification
        return F.log_softmax(x, dim=1)


def load_cora(path="./cora", device="cpu"):
    """
    The graph convolutional operation rquires the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2). This step
    scales the adjacency matrix such that the features of neighboring nodes are weighted appropriately during
    aggregation. The steps involved in the renormalization trick are as follows:
        - Compute the degree matrix.
        - Compute the inverse square root of the degree matrix.
        - Multiply the inverse square root of the degree matrix with the adjacency matrix.
    """

    # Set the paths to the data files
    content_path = os.path.join(path, "cora.content")
    cites_path = os.path.join(path, "cora.cites")

    # Load data from files
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    # Process features
    features = torch.FloatTensor(
        content_tensor[:, 1:-1].astype(np.int32)
    )  # Extract feature values
    scale_vector = torch.sum(features, dim=1)  # Compute sum of features for each node
    scale_vector = 1 / scale_vector  # Compute reciprocal of the sums
    scale_vector[scale_vector == float("inf")] = 0  # Handle division by zero cases
    scale_vector = torch.diag(
        scale_vector
    ).to_sparse()  # Convert the scale vector to a sparse diagonal matrix
    features = scale_vector @ features  # Scale the features using the scale vector

    # Process labels
    classes, labels = np.unique(
        content_tensor[:, -1], return_inverse=True
    )  # Extract unique classes and map labels to indices
    labels = torch.LongTensor(labels)  # Convert labels to a tensor

    # Process adjacency matrix
    idx = content_tensor[:, 0].astype(np.int32)  # Extract node indices
    idx_map = {
        id: pos for pos, id in enumerate(idx)
    }  # Create a dictionary to map indices to positions

    # Map node indices to positions in the adjacency matrix
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], cites_tensor)),
        dtype=np.int32,
    )

    V = len(idx)  # Number of nodes
    E = edges.shape[0]  # Number of edges
    adj_mat = torch.sparse_coo_tensor(
        edges.T, torch.ones(E), (V, V), dtype=torch.int64
    )  # Create the initial adjacency matrix as a sparse tensor
    adj_mat = torch.eye(V) + adj_mat  # Add self-loops to the adjacency matrix

    # MINE: following operations fail with torch 1.13.1 (required by ifbo)
    # degree_mat = torch.sum(adj_mat, dim=1) # Compute the sum of each row in the adjacency matrix (degree matrix)
    # degree_mat = torch.sqrt(1 / degree_mat) # Compute the reciprocal square root of the degrees
    # degree_mat[degree_mat == float('inf')] = 0 # Handle division by zero cases
    # degree_mat = torch.diag(degree_mat).to_sparse() # Convert the degree matrix to a sparse diagonal matrix

    # adj_mat = degree_mat @ adj_mat @ degree_mat # Apply the renormalization trick

    return (
        features.to_sparse().to(device),
        labels.to(device),
        adj_mat.to_sparse().to(device),
    )


def train_epoch(model, optimizer, criterion, input, target, mask_train, mask_val):
    model.train()
    optimizer.zero_grad()
    output = model(*input)
    loss = criterion(output[mask_train], target[mask_train])
    loss.backward()
    optimizer.step()

    val_acc, val_err, val_loss = evaluate(model, criterion, input, target, mask_val)
    return val_acc, val_err, val_loss


def evaluate(model, criterion, input, target, mask):
    set_seed()
    model.eval()
    with torch.no_grad():
        output = model(*input)
        val_loss = criterion(output[mask], target[mask])
        val_acc = (output[mask].argmax(dim=1) == target[mask]).sum().item() / len(
            target[mask]
        )
        val_err = 1 - val_acc
    return val_acc, val_err, val_loss


def run_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    epoch=50,  # 10 default if not handled by the searcher
    hidden_dim=16,
    dropout_p=0.5,
    include_bias=False,
):
    start = time.time()
    # for mf algorithms
    epochs = int(epoch)
    features, labels, adj_mat = load_cora(device=device)
    idx = torch.randperm(len(labels)).to(device)
    # test:         0 - 999
    # validation:   1000 - 1499
    # train:        1500 - end
    idx_test, idx_val, idx_train = idx[:1000], idx[1000:1500], idx[1500:]

    model = GCN(
        features.shape[1], hidden_dim, labels.max().item() + 1, include_bias, dropout_p
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon
    )
    criterion = nn.NLLLoss()

    # checkpointing to resume model training in higher fidelities
    previous_state = load_checkpoint(
        directory=previous_pipeline_directory,
        model=model,
        optimizer=optimizer,
    )

    if previous_state is not None:
        start_epoch = previous_state["epochs_trained"]
    else:
        start_epoch = 0

    val_losses = list()
    test_losses = list()

    for ep in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        val_acc, val_error, val_loss = train_epoch(
            model, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val
        )
        val_losses.append(val_loss.item())
        test_acc, test_error, test_loss = evaluate(
            model, criterion, (features, adj_mat), labels, idx_test
        )
        test_losses.append(test_loss.item())

    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        },
    )
    print(f"  Epoch {epochs} / {epochs} Val Loss: {val_loss}".ljust(2))
    end = time.time()

    learning_curves, min_valid_seen, min_test_seen = process_trajectory(
        pipeline_directory, val_loss, val_losses, test_losses, test_loss
    )

    return {
        "cost": epochs - start_epoch,
        "info_dict": {
            "continuation_fidelity": None,
            "cost": epochs - start_epoch,
            "end_time": end,
            "fidelity": epochs,
            "learning_curve": val_losses,
            "learning_curves": learning_curves,
            "max_fidelity_cost": epochs,
            "max_fidelity_loss": val_losses[-1],
            # "min_test_ever": np.min(test_losses),
            "min_test_seen": np.min(learning_curves["test"]),
            # "min_valid_ever": np.min(val_losses),
            "min_valid_seen": np.min(learning_curves["valid"]),
            "process_id": os.getpid(),
            "start_time": start,
            "test_score": test_loss,
            "val_score": -val_loss,
        },
        "loss": val_loss,
    }