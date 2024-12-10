#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np

def build_graph(train_data):
    """
    Builds a directed graph (DiGraph) from the training data.

    :param train_data: List of user-item interactions sequences.
    :return: A NetworkX DiGraph with edges representing user-item interactions.
    """
    graph = nx.DiGraph() # Create an empty directed graph
    for seq in train_data:  # For each sequence, add edges between consecutive items
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None: # If the edge already exists, increment its weight, otherwise initialize with weight 1
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight) # Add the edge with the weight
    # Normalize the weights for incoming edges for each node
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph

def data_masks(all_usr_pois, item_tail):
    """
    Creates masks for the user-item interactions and pads sequences with item_tail.

    :param all_usr_pois: List of user-item sequences.
    :param item_tail: Value to pad sequences with, generally the last item in the sequence.
    :return: Tuple containing padded sequences, corresponding masks, and maximum sequence length.
    """
    us_lens = [len(upois) for upois in all_usr_pois] # Calculate the length of each sequence
    len_max = max(us_lens)  # Find the maximum length of the sequences
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] # Create mask with 1s for real items, 0 for padding
    return us_pois, us_msks, len_max # Return padded sequences, masks, and the maximum sequence length

def split_validation(train_set, valid_portion):
    """
    Splits the dataset into training and validation sets.

    :param train_set: Tuple containing training data (input sequences and target labels).
    :param valid_portion: Proportion of data to be used for validation.
    :return: Tuple containing training and validation sets.
    """
    train_set_x, train_set_y = train_set # Separate input data and targets
    n_samples = len(train_set_x)  # Number of samples in the training set
    sidx = np.arange(n_samples, dtype='int32')  # Create an array of sample indices
    np.random.shuffle(sidx) # Shuffle the indices for random splitting
    n_train = int(np.round(n_samples * (1. - valid_portion))) # Calculate the number of training samples
    # Split the data into training and validation sets based on the shuffled indices
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    """
    Class for managing dataset, including data loading, batching, and slicing.
    """
    def __init__(self, data, shuffle=False, graph=None, opt=None):
        """
        Initializes the Data class by preparing the input data, masks, and targets.

        :param data: Tuple containing input sequences and target labels.
        :param shuffle: Boolean indicating whether to shuffle the data.
        :param graph: Optional pre-built graph to associate with the data.
        :param opt: Optional parameters, such as whether the targets are dynamic.
        """
        inputs = data[0]  # Input user-item sequences
        inputs, mask, len_max = data_masks(inputs, [0])  # Generate masks and padded inputs
        self.inputs = np.asarray(inputs)  # Convert inputs to NumPy array
        self.mask = np.asarray(mask)  # Convert masks to NumPy array
        self.len_max = len_max  # Store the maximum sequence length
        self.targets = np.asarray(data[1])  # Store target labels
        if opt.dynamic:
            self.targets = np.asarray(data[2])  # If dynamic targets are specified, update the targets
        self.length = len(inputs)  # Store the number of sequences in the dataset
        self.shuffle = shuffle  # Flag for shuffling the data
        self.graph = graph  # Store the optional graphh

    def generate_batch(self, batch_size):
        """
        Generates batches of data, optionally shuffling the data.

        :param batch_size: The size of each batch.
        :return: List of batch indices for the dataset.
        """
        if self.shuffle:
            shuffled_arg = np.arange(self.length)  # Create an array of indices
            np.random.shuffle(shuffled_arg)  # Shuffle the indices
            self.inputs = self.inputs[shuffled_arg]  # Shuffle the input sequences
            self.mask = self.mask[shuffled_arg]  # Shuffle the masks
            self.targets = self.targets[shuffled_arg]  # Shuffle the target labels
        n_batch = int(self.length / batch_size)  # Calculate the number of batches
        if self.length % batch_size != 0:
            n_batch += 1  # If there are leftover samples, add an extra batch
        slices = np.split(np.arange(n_batch * batch_size), n_batch)  # Split indices into batches
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]  # Adjust the last batch to include all samples
        return slices

    def get_slice(self, i):
        """
        Returns the i-th slice of data, including inputs, masks, and targets.

        :param i: The index of the slice.
        :return: Tuple of alias inputs, adjacency matrix, items, mask, and targets.
        """
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]  # Get the slice of inputs, mask, and targets
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input))) # Count unique nodes in each input sequence
        max_n_node = np.max(n_node) # Find the maximum number of unique nodes
        for u_input in inputs:
            node = np.unique(u_input)  # Get unique nodes in the input sequence
            items.append(node.tolist() + (max_n_node - len(node)) * [0])  # Pad items to the maximum number of nodes
            u_A = np.zeros((max_n_node, max_n_node))  # Initialize the adjacency matrix for the nodes
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]  # Get the index of the current node
                v = np.where(node == u_input[i + 1])[0][0]  # Get the index of the next node
                u_A[u][v] = 1  # Set the corresponding edge to 1
            u_sum_in = np.sum(u_A, 0)  # Calculate the sum of incoming edges
            u_sum_in[np.where(u_sum_in == 0)] = 1  # Avoid division by zero
            u_A_in = np.divide(u_A, u_sum_in)  # Normalize the incoming edges
            u_sum_out = np.sum(u_A, 1)  # Calculate the sum of outgoing edges
            u_sum_out[np.where(u_sum_out == 0)] = 1  # Avoid division by zero
            u_A_out = np.divide(u_A.transpose(), u_sum_out)  # Normalize the outgoing edges
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()  # Concatenate and transpose the matrices
            A.append(u_A)  # Store the adjacency matrix for the user input
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])  # Store the alias inputs
        return alias_inputs, A, items, mask, targets