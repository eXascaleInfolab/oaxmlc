import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from tqdm import tqdm
import networkx as nx
import itertools
import collections


def print_time(str):
    print(str, '--', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def format_time_diff(start_time, stop_time):
    total_time = stop_time - start_time
    return f"{int(total_time / 3600)} hours {int((total_time % 3600) / 60)} minutes {total_time % 60:.1f} seconds"


# Print some info about the CUDA memory currently used
def print_memory_info(cfg):
    print(f">>> Total memory: {torch.cuda.get_device_properties(cfg['device']).total_memory / 1024**3} GB")
    print(f">>> Reserved memory: {torch.cuda.memory_reserved(cfg['device']) / 1024**3} GB")
    print(f">>> Allocated memory: {torch.cuda.memory_allocated(cfg['device']) / 1024**3} GB")


# Print the config
def print_config(cfg):
    print()
    print(f"> Config")
    for key, value in cfg.items():
        if key in ['paths', 'tasks_size', 'task_to_subroot', 'label_to_tasks']: continue
        if key == 'tamlec_params':
            for subkey, val in value.items():
                if subkey in ['abstract_dict', 'trg_vocab', 'src_vocab', 'taxos_hector', 'taxos_tamlec']: continue
                print(f">> tamlec_{subkey}: {val}")
        else:
            print(f">> {key}: {value}")
    print()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_histogram(data, path, title):
    sns.set_style("whitegrid")
    ratio = 0.5
    plt.figure(figsize=(16*ratio,9*ratio))
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    min_val = np.min(data)
    max_val = np.max(data)
    # sns.histplot(data=data, color='r', binwidth=1)
    sns.histplot(data=data, color='r', bins=int(max_val-min_val), cumulative=True)
    plt.title(f"{title} - mean {mean:.2f} - std {std:.2f} - min {min_val:.2f} - max {max_val:.2f}")
    # plt.title(f"{title} - mean {mean:.2f}")
    # plt.yscale('log')
    # plt.xlabel("Number of labels")
    # plt.ylabel("Number of documents")
    plt.savefig(path, bbox_inches='tight')


# Compute some stats on a preprocessed dataset
def compute_preproc_dataset_stats(output_path, dataloaders):
    stats_df = pd.DataFrame(columns=['split', 'docs', 'labels', 'avg_lab/doc', 'med_lab/doc', 'avg_doc/lab', 'med_doc/lab'])
    splits = ['train', 'validation', 'test']
    for split in splits:
        lab_per_doc = torch.tensor([])
        doc_per_lab = None
        docs = []
        for input_data, labels, label_mask in tqdm(dataloaders[f"global_{split}"], leave=False):
            if doc_per_lab is None: doc_per_lab = torch.zeros(label_mask.size(1))
            labels = torch.gather(labels, dim=1, index=label_mask)
            docs.append(len(input_data))
            d_p_l = torch.sum(labels, dim=0)
            doc_per_lab += d_p_l
            l_p_d = torch.sum(labels, dim=1)
            lab_per_doc = torch.concat([lab_per_doc, l_p_d], dim=0)
        dict_stats = pd.DataFrame({
            'split': split,
            'docs': sum(docs),
            'labels': labels.size(1),
            'avg_lab/doc': torch.round(torch.mean(lab_per_doc), decimals=2).item(),
            'med_lab/doc': torch.round(torch.median(lab_per_doc), decimals=2).item(),
            'avg_doc/lab': torch.round(torch.mean(doc_per_lab), decimals=2).item(),
            'med_doc/lab': torch.round(torch.median(doc_per_lab), decimals=2).item(),
        }, index=[0])

        torch.save(lab_per_doc.numpy(), output_path / f"{split}_lab_per_doc.pt")
        torch.save(doc_per_lab.numpy(), output_path / f"{split}_doc_per_lab.pt")
        stats_df = pd.concat([stats_df, dict_stats], ignore_index=True)
    stats_df.to_markdown(output_path / f"preproc_dataset_stats.md", index=False)


def compute_n_docs(output_path, dataloaders):
    n_docs = {}
    splits = ['train', 'validation', 'test']
    for split in splits:
        n_docs[split] = {}
        for task_id, (batched_input, batched_labels, column_indices) in enumerate(tqdm(dataloaders[f"tasks_{split}"], leave=False)):
            n_docs[split][task_id] = {}
            full_labels = torch.vstack(batched_labels)
            full_labels = full_labels[:, column_indices]
            n_labels = torch.sum(full_labels, dim=1)
            for k in [1,2,3,4,5]:
                n_doc = torch.sum(n_labels >= k).item()
                n_docs[split][task_id][f"@{k}"] = n_doc

    torch.save(n_docs, f"{output_path}/num_docs.pt")

    with open(f"{output_path}/num_docs.txt", 'w') as f:
        for split in splits:
            f.write(f"{split}\n")
            for task, my_dict in n_docs[split].items():
                f.write(f"Task {task} -> {my_dict}\n")
            f.write('\n')


# Draw the taxonomy of a given task and dataset
def draw_tasks(taxos_tamlec, output_path, dataset_name):
    print(f"> Draw tasks...")
    for task_id in range(len(taxos_tamlec)):
        root_node = taxos_tamlec[task_id][0]
        # Construct edge list of the graph
        all_edges = []
        for parent, children in taxos_tamlec[task_id][1].items():
            for child in children:
                all_edges.append((parent, child))

        # Create the graph, draw and save it
        G = nx.DiGraph()
        G.add_edges_from(all_edges)
        pos = semilattice_pos(G, root_node)
        plt.figure(figsize=(16,9))
        nx.draw(G, pos=pos, with_labels=True, node_size=900, node_color='red', linewidths=3)
        plt.savefig(output_path / f"{dataset_name}_task{task_id}.png")
        plt.close()


# Draw a semi-lattice in a pretty way with networkx
# https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
def semilattice_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    A layout function for plotting a semi-lattice (a directed acyclic graph where nodes may have multiple parents).
    G: The graph (must be a directed acyclic graph).
    root: The root node of the layout. If not provided, a root will be chosen.
    width: Horizontal space allocated for the layout.
    vert_gap: Gap between levels of the hierarchy.
    vert_loc: Vertical position of the root.
    xcenter: Horizontal position of the root.
    '''
    if not nx.is_directed_acyclic_graph(G):
        raise TypeError('The function only supports directed acyclic graphs.')

    if root is None:
        # Attempt to find a root-like node (with no incoming edges)
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if not roots:
            raise ValueError('The graph has no root-like nodes.')
        root = roots[0]

    def _calculate_positions(G, node, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, visited=None):
        '''
        Helper function to recursively calculate positions for nodes.
        
        pos: A dictionary storing positions of nodes.
        visited: A set to track visited nodes and avoid processing them multiple times.
        '''
        if pos is None:
            pos = {}
        if visited is None:
            visited = set()

        if node in visited:
            return pos  # Avoid reprocessing nodes

        visited.add(node)
        pos[node] = (xcenter, vert_loc)

        children = list(G.successors(node))
        if children:
            dx = width / len(children)
            next_x = xcenter - width / 2 + dx / 2
            for child in children:
                pos = _calculate_positions(G, child, width=dx, vert_gap=vert_gap, 
                                            vert_loc=vert_loc - vert_gap, xcenter=next_x, 
                                            pos=pos, visited=visited)
                next_x += dx

        return pos

    # Start recursive position calculation
    pos = _calculate_positions(G, root, width=width, vert_gap=vert_gap, 
                                vert_loc=vert_loc, xcenter=xcenter)
    
    return pos


class LRHandler:
    """Handles learning rate scheduling with warm-up, cosine decay, and a minimum learning rate threshold.

    This class adjusts the learning rate of an optimizer dynamically over the course of training. 
    It supports a warm-up phase at the start of training, followed by a cosine decay schedule, 
    and maintains a minimum learning rate after the decay phase.

    Args:
        cfg (dict): Configuration dictionary.
        optimizer (torch.optim.Optimizer): The optimizer for which learning rate will be updated.

    Attributes:
        epochs (int): Total number of epochs.
        max_lr (float): Maximum learning rate.
        warm_up_epochs (int): Number of warm-up epochs (20% of total epochs).
        decay_epochs (int): Number of decay epochs (80% of total epochs).
        min_lr (float): Minimum learning rate (10% of the maximum learning rate).
        optimizer (torch.optim.Optimizer): The optimizer for which learning rate will be updated.
        curr_lr (float): Current learning rate being used.

    Methods:
        update_lr(curr_epoch):
            Updates the learning rate based on the current epoch.
    """

    def __init__(self, cfg, optimizer):
        self.epochs = cfg['epochs']
        self.max_lr = cfg['learning_rate']
        self.warm_up_epochs = int(0.2 * self.epochs)
        self.decay_epochs = int(0.8 * self.epochs)
        self.min_lr = 0.1 * self.max_lr
        self.optimizer = optimizer

    def update_lr(self, curr_epoch):
        # Update the lr according the curr_epoch
        # Warm-up phase
        if curr_epoch <= self.warm_up_epochs:
            self.curr_lr = curr_epoch * self.max_lr / self.warm_up_epochs

        # After the decay, keep to the min_lr
        elif curr_epoch > self.decay_epochs:
            self.curr_lr = self.min_lr

        # Cosine decay
        else:
            decay_ratio = (curr_epoch - self.warm_up_epochs) / (self.decay_epochs - self.warm_up_epochs)
            self.curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * decay_ratio))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.curr_lr


class EarlyStopping:
    """
    Implements early stopping to monitor training and stop the process when the performance metric does not improve within a specified patience.

    Attributes:
        max_patience (int): Maximum number of epochs to wait for improvement before stopping.
        cfg (dict): Configuration dictionary.
        best_metric (float): Best metric value achieved during training.
        current_patience (int): Number of epochs since the last improvement.
        metrics_dict (dict): Stores the metrics of the best model so far.
        best_epoch (int): Epoch at which the best metric was achieved.
        method_to_min_epoch (dict): Maps algorithms to their minimum number of epoch before starting the monitoring.
        min_epoch (int): Minimum number of epochs required before early stopping is applied.

    Methods:
        checkpoint(new_model, metrics_dict, epoch): Evaluates the current epoch and decides whether to stop or continue.
        _save_model(new_model): Saves the current model to the configured path.
    """

    def __init__(self, max_patience, cfg):
        self.cfg = cfg
        self.best_metric = float('inf')
        self.max_patience = max_patience
        self.current_patience = 0
        self.metrics_dict = {}
        self.best_epoch = 0
        self.method_to_min_epoch = {
            'match': 5,
            'xmlcnn': 5,
            'attentionxml': 5,
            'hector': 5,
            'tamlec': 5,
            'lightxml': 5,
            'cascadexml': 5,
        }
        self.min_epoch = self.method_to_min_epoch[self.cfg['method']]

    def checkpoint(self, new_model, metrics_dict, epoch):
        # Minimum number of epoch on which we do not check
        if epoch < self.min_epoch:
            # Anyway still save the model in training for the first epochs
            self._save_model(new_model)
            return True

        # Take last loss and check if improving or not
        new_metric = metrics_dict['loss']
        if new_metric < self.best_metric:
            # Save the model
            self._save_model(new_model)
            self.metrics_dict = metrics_dict
            self.best_epoch = epoch
            self.best_metric = new_metric
            self.current_patience = 0
        else:
            self.current_patience += 1

        if self.current_patience == self.max_patience:
            return False
        return True

    def _save_model(self, new_model):
        if self.cfg['method'] in ['hector', 'tamlec', 'attentionxml']:
            torch.save(new_model, self.cfg['paths']['model'])
        else:
            torch.save(copy.deepcopy(new_model.state_dict()), self.cfg['paths']['state_dict'])


class AdaptivePatience:
    """
    Manages adaptive patience for multi-task learning, allowing independent early stopping
    for individual tasks as well as a global criterion.

    Attributes:
        max_patience (int): Maximum patience for all tasks and the global criterion.
        cfg (dict): Configuration dictionary.
        n_tasks (int): Total number of tasks being trained.
        tasks_to_complete (list): List of task IDs that are yet to complete training.
        patiences (dict): Dictionary of `EarlyStopping` objects for each task and global criterion.
        tasks_completed (list): List of task IDs that have completed training.
        metrics_dicts (dict): Stores the best metrics for completed tasks.
        best_epochs (dict): Tracks the epoch at which each task achieved its best performance.
    
    Methods:
        global_checkpoint(new_model, metrics_dict, epoch): Checks global and task-specific criteria for early stopping.
        tasks_checkpoint(new_model, metrics_dict, epoch): Checks task-specific criteria for early stopping.
        mark_task_done(task_id): Marks a task as completed manually.
    """

    def __init__(self, max_patience, cfg, n_tasks):
        self.cfg = cfg
        self.max_patience = max_patience
        self.n_tasks = n_tasks
        self.tasks_to_complete = list(range(n_tasks))
        self.patiences = {idx: EarlyStopping(self.max_patience, self.cfg) for idx in self.tasks_to_complete}
        self.patiences[self.cfg['all_tasks_key']] = EarlyStopping(self.max_patience, self.cfg)
        self.tasks_completed = []
        self.metrics_dicts = {}
        self.best_epochs = {}

    def global_checkpoint(self, new_model, metrics_dict, epoch):
        # Check if any task has converged
        for task_id in self.tasks_to_complete:
            if not self.patiences[task_id].checkpoint(new_model, metrics_dict[task_id], epoch):
                self.tasks_completed.append(task_id)
                self.tasks_to_complete.remove(task_id)
                self.metrics_dicts[task_id] = metrics_dict[task_id]
                self.best_epochs[task_id] = epoch
                print(f">>> Task {task_id} completed at epoch {self.best_epochs[task_id]}, still to complete: {self.tasks_to_complete}")

        # If converged on the global loss then stop, otherwise continue training
        return self.patiences[self.cfg['all_tasks_key']].checkpoint(new_model, metrics_dict[self.cfg['all_tasks_key']], epoch)

    def tasks_checkpoint(self, new_model, metrics_dict, epoch):
        # Check if any task has converged, and stop it if this is the case
        for task_id in self.tasks_to_complete:
            if not self.patiences[task_id].checkpoint(new_model, metrics_dict[task_id], epoch):
                self.tasks_completed.append(task_id)
                self.tasks_to_complete.remove(task_id)
                self.metrics_dicts = metrics_dict[task_id]
                self.best_epochs[task_id] = epoch
                print(f">>> Task {task_id} completed at epoch {self.best_epochs[task_id]}, still to complete: {self.tasks_to_complete}")
        return len(self.tasks_to_complete) != 0

    def mark_task_done(self, task_id):
        if task_id not in self.tasks_completed:
            self.tasks_to_complete.remove(task_id)
            self.tasks_completed.append(task_id)


class MetricsHandler:
    """
    Manages metrics during training or evaluation by storing them in a pandas.DataFrame and exporting to a CSV file.

    Attributes:
        df (pandas.DataFrame): DataFrame to store metrics, initialized with specified columns.
        output_path (str): Path to the CSV file where metrics will be saved.

    Methods:
        add_row(new_row_dict): Adds a new row of metrics to the DataFrame and updates the CSV file.
    """

    def __init__(self, columns, output_path):
        self.df = pd.DataFrame(columns=columns)
        self.output_path = output_path

    def add_row(self, new_row_dict):
        assert isinstance(new_row_dict, dict), f"new_row_dict argument should be a Python dictionary"
        new_row = pd.DataFrame(new_row_dict, index=[0])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.output_path, float_format=str)
