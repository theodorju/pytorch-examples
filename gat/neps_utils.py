import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neps.utils.common import load_checkpoint, save_checkpoint
from neps_global_utils import set_seed, process_trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

        This operation can be mathematically described as:

            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate

        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        #  Initialize the weight matrix W 
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))

        # Initialize the attention weights a
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients

        self.reset_parameters() # Reset the parameters


    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
    

    def _get_attention_scores(self, h_transformed: torch.Tensor):
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

        # broadcast add 
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        n_nodes = h.shape[0]

        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        h_transformed = torch.mm(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        
        # getting the attention scores
        # output shape (n_heads, n_nodes, n_nodes)
        e = self._get_attention_scores(h_transformed)

        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
        
        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # final node embeddings are computed as a weighted average of the features of its neighbors
        h_prime = torch.matmul(attention, h_transformed)

        # concatenating/averaging the attention heads
        # output shape (n_nodes, out_features)
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)

        return h_prime

################################
### MAIN GAT NETWORK MODULE  ###
################################

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    Consists of a 2-layer stack of Graph Attention Layers (GATs). The fist GAT Layer is followed by an ELU activation.
    And the second (final) layer is a GAT layer with a single attention head and softmax activation function. 
    """
    def __init__(self,
        in_features,
        n_hidden,
        n_heads,
        num_classes,
        concat=False,
        dropout=0.4,
        leaky_relu_slope=0.2):
        """ Initializes the GAT model. 

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """

        super(GAT, self).__init__()

        # Define the Graph Attention layers
        self.gat1 = GraphAttentionLayer(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        
        self.gat2 = GraphAttentionLayer(
            in_features=n_hidden, out_features=num_classes, n_heads=1,
            concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        

    def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):
        """
        Performs a forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """

        # Apply the first Graph Attention layer
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x) # Apply ELU activation function to the output of the first layer

        # Apply the second Graph Attention layer
        x = self.gat2(x, adj_mat)

        return F.log_softmax(x, dim=1) # Apply log softmax activation function


def load_cora(path='./cora', device='cpu'):
    """
    Loads the Cora dataset. The dataset is downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz.

    """

    # Set the paths to the data files
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    # Load data from files
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    # Process features
    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32)) # Extract feature values
    scale_vector = torch.sum(features, dim=1) # Compute sum of features for each node
    scale_vector = 1 / scale_vector # Compute reciprocal of the sums
    scale_vector[scale_vector == float('inf')] = 0 # Handle division by zero cases
    scale_vector = torch.diag(scale_vector).to_sparse() # Convert the scale vector to a sparse diagonal matrix
    features = scale_vector @ features # Scale the features using the scale vector

    # Process labels
    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True) # Extract unique classes and map labels to indices
    labels = torch.LongTensor(labels) # Convert labels to a tensor

    # Process adjacency matrix
    idx = content_tensor[:, 0].astype(np.int32) # Extract node indices
    idx_map = {id: pos for pos, id in enumerate(idx)} # Create a dictionary to map indices to positions

    # Map node indices to positions in the adjacency matrix
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)

    V = len(idx) # Number of nodes
    E = edges.shape[0] # Number of edges
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64) # Create the initial adjacency matrix as a sparse tensor
    adj_mat = torch.eye(V) + adj_mat # Add self-loops to the adjacency matrix

    # return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)
    return features.to(device), labels.to(device), adj_mat.to(device)


def train_epoch(model, optimizer, criterion, input, target, mask_train, mask_val): 
    model.train()
    optimizer.zero_grad()
    output = model(*input)
    loss = criterion(output[mask_train], target[mask_train])
    loss.backward()
    optimizer.step()

    val_acc, val_err, val_loss = evaluate(model, criterion, input, target, mask_val)
    return val_acc, val_err, val_loss

def evaluate(model, criterion, input, target, mask_val):
    set_seed()
    model.eval()
    with torch.no_grad():
        output = model(*input)
        val_loss = criterion(output[mask_val], target[mask_val])
        val_acc = (output[mask_val].argmax(dim=1) == target[mask_val]).sum().item() / len(target[mask_val])
        val_err = 1 - val_acc
        return val_acc, val_err, val_loss


def run_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    epoch=300,  # 300 default if not handled by the searcher
    n_hidden=64,
    dropout=0.6,
    leaky_relu_slope=0.2,
    concat_heads=False,
    n_heads=8,
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

    model = GAT(
        in_features=features.shape[1], n_hidden=n_hidden, n_heads=n_heads, num_classes=labels.max().item() + 1, concat=concat_heads, dropout=dropout, leaky_relu_slope=leaky_relu_slope
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

    val_errors = list()

    for ep in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(ep + 1, epochs).ljust(2))
        val_acc, val_error, val_loss = train_epoch(
            model, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val
        )
        val_errors.append(val_error)
    test_acc, test_error, test_loss = evaluate(
        model, criterion, (features, adj_mat), labels, idx_test
    )

    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optimizer,
        values_to_save={
            "epochs_trained": epochs,
        },
    )
    print(
        "  Epoch {} / {} Val Loss: {}".format(epochs, epochs, val_loss).ljust(2)
    )
    end = time.time()

    learning_curves, min_valid_seen, min_test_seen = process_trajectory(
        pipeline_directory, val_loss, test_loss
    )
    # random search - no fidelity hyperparameter
    if "random_search" in str(pipeline_directory) or "hyperband" in str(
        pipeline_directory
    ):
        return {
            "loss": val_loss,
            "info_dict": {
                "test_accuracy": test_acc,
                "val_errors": val_errors,
                "train_time": end - start,
                "cost": epochs - start_epoch,
            },
            "cost": epochs - start_epoch,
        }

    return {
        "loss": val_loss,  # validation loss
        "cost": epochs - start_epoch,
        "info_dict": {
            "cost": epochs - start_epoch,
            "val_score": -val_loss,  # - validation loss for this fidelity
            "test_score": test_loss,  # test loss (w/out minus)
            "fidelity": epochs,
            "continuation_fidelity": None,  # keep None
            "start_time": start,
            "end_time": end,
            "max_fidelity_loss": None,
            "max_fidelity_cost": None,
            "min_valid_seen": min_valid_seen,
            "min_test_seen": min_test_seen,
            # "min_valid_ever": None,       # Cannot calculate for real datasets
            # "min_test_ever": None,          # Cannot calculate for real datasets
            "learning_curve": val_loss,  # validation loss (w/out minus)
            "learning_curves": learning_curves,  # dict: valid: [..valid_losses..], test: [..test_losses..], fidelity: [1, 2, ...]
        },
    }