import random
import torch
from torch.utils.data import Sampler
import numpy as np
import copy
from scipy.sparse import csr_matrix
from tqdm import tqdm


class SubtreeSampler(Sampler):
    """SubtreeSampler Class

    This class is a custom data sampler designed for tasks involving hierarchical taxonomies. It manages the sampling of subtrees, handles the creation of support and query sets for protonet and maml algorithms, and implements various collation strategies tailored to different algorithms.

    Attributes:
        cfg (dict): Configuration dictionary
        dataset (list): List of subtrees, where each subtree is represented as a tuple with:
            - torch.tensor: tokenized documents.
            - torch.tensor: labels.
            - torch.tensor: Column indices of relevant labels.
        batch_size (int): Number of examples in a batch.
        possible_subtrees (list): List of subtree indices.
        items_idx_per_label (dict): Mapping of subtree indices to dictionaries that map labels to sets of item indices.

    Methods:
        __len__():
            Returns the number of subtrees in the dataset.

        __iter__():
            Iterates through the subtrees, yielding subtree indices.

        collate_standard(seed=None):
            Generates a closure for standard collation, creating mini-batches of inputs and labels.

        collate_fastxml(input_data):
            Collates data for FastXML, transforming inputs into sparse matrices and grouping labels.

        collate_hector_tamlec(seed=None):
            Generates a closure for Hector/Tamlec collation, creating mini-batches while maintaining taxonomy-specific constraints.
    """

    def __init__(self, dataset, cfg, batch_size):
        super().__init__(data_source=None)
        self.cfg = cfg
        self.taxonomy = self.cfg['taxonomy']
        self.dataset = dataset
        self.batch_size = batch_size

        self.possible_subtrees = list(range(len(self.dataset)))
        if self.cfg['method'] in ['protonet', 'bdc', 'maml']:
            # key: subtree index, values: dict with labels as key and set of items index having this label as values
            self.items_idx_per_label = {}
            for subtree_idx in tqdm(self.possible_subtrees, leave=False):
                self.items_idx_per_label[subtree_idx] = {}
                column_indices = self.dataset[subtree_idx][2]
                relevant_labels = self.dataset[subtree_idx][1][:, column_indices]
                # Get all labels from all items
                for i, col_idx in enumerate(column_indices):
                    col_idx = col_idx.item()
                    column = relevant_labels[:,i]
                    indices = torch.nonzero(column, as_tuple=True)[0]
                    for index in indices:
                        if col_idx in self.items_idx_per_label[subtree_idx]:
                            self.items_idx_per_label[subtree_idx][col_idx].add(index.item())
                        else:
                            self.items_idx_per_label[subtree_idx][col_idx] = {index.item()}


    def __len__(self):
        return len(self.possible_subtrees)


    def __iter__(self):
        for subtree_idx in self.possible_subtrees:
            self.subtree_idx = subtree_idx
            self.sampled_labels = self.dataset[subtree_idx][2].tolist()

            yield subtree_idx


    def collate_standard(self, seed=None):
        def _collate_standard(input_data):
            """Collate function for standard classification tasks. The seed can be defined in the closure so that in training we have random batches while in evaluation we have always the same batches.

            Args:
                input_data (list): A list of one tuple containing three elements:
                    - A torch.tensor containing the inputs for the model.
                    - A torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

            Returns:
                tuple: A tuple containing three elements:
                    - A list of torch.tensor containing the inputs for the model.
                    - A list of torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
            """

            document_data = input_data[0][0]
            labels_data = input_data[0][1]
            column_indices = input_data[0][2]

            # Create mini-batches
            indices = np.arange(len(document_data))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            n_batches = int(np.ceil(len(indices)/self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, n_batches)]
            # Remove empty batch if any
            batches_indices = [batch for batch in batches_indices if len(batch) != 0]
            # Create batched input and labels
            batched_input = [document_data[batch] for batch in batches_indices]
            batched_labels = [labels_data[batch] for batch in batches_indices]

            return (
                batched_input,
                batched_labels,
                column_indices,
            )
        return _collate_standard


    def collate_no_batch(self, input_data):
        """Collate function that returns the entire dataset with no batches.

        Args:
            input_data (list): A list of one tuple containing three elements:
                - A torch.tensor containing the inputs for the model.
                - A list of lists containing the labels.
                - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

        Returns:
            tuple: A tuple containing three elements:
                - A torch.tensor containing the inputs for the model.
                - A list of lists containing the labels.
                - A list of lists with relevant column indices, i.e. labels that appear in this subtree.
        """

        document_data = input_data[0][0]
        labels_data = input_data[0][1]
        column_indices = input_data[0][2]

        return (
            document_data,
            labels_data,
            column_indices,
        )


    def collate_hector_tamlec(self, seed=None):
        def _collate_hector_tamlec(input_data):
            """Collate function for Hector and Tamlec. The seed can be defined in the closure so that in training we have random batches while in evaluation we have always the same batches.

            Args:
                input_data (list): A list of one tuple containing four elements:
                    - A torch.tensor containing the inputs for the model.
                    - A list of lists containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
                    - An integer representing the task id

            Returns:
                tuple: A tuple containing three elements:
                    - A list of torch.tensor containing the inputs for the model.
                    - A list of lists of lists containing the labels.
                    - A list with relevant column indices, i.e. labels that appear in this subtree.
            """

            document_data = input_data[0][0]
            labels_data = input_data[0][1]
            column_indices = input_data[0][2]
            task_id = input_data[0][3]

            # Create mini-batches
            indices = np.arange(len(document_data))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            n_batches = int(np.ceil(len(indices)/self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, n_batches)]
            # Remove empty batch if any
            batches_indices = [batch for batch in batches_indices if len(batch) != 0]
            # Create batched input and labels
            batched_input = []
            batched_labels = []
            if self.cfg['method'] == 'tamlec':
                for batch in batches_indices:
                    # Use slicing since document_data is a tensor
                    batched_input.append(document_data[batch])
                    # Labels: keep all labels appearing in the sub-tree (root included since tamlec requires the start of the path)
                    labels_batch = []
                    for idx in batch:
                        labels_batch.append([self.cfg['task_to_subroot'][task_id]] + [lab for lab in labels_data[idx] if lab in column_indices])
                    batched_labels.append(labels_batch)
            else:
                for batch in batches_indices:
                    # Use slicing since document_data is a tensor
                    batched_input.append(document_data[batch])
                    # Labels: keep root of the taxonomy and of the sub-tree (since hector requires the start of the path)
                    labels_batch = []
                    for idx in batch:
                        labels_batch.append([0] + [self.cfg['task_to_subroot'][task_id]] + [lab for lab in labels_data[idx] if lab in column_indices])
                    batched_labels.append(labels_batch)

            return (
                batched_input,
                batched_labels,
                column_indices,
            )
        return _collate_hector_tamlec
