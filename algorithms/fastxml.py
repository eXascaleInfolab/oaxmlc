from tqdm import tqdm
import torch
import numpy as np
import time
import warnings

from datahandler.dataloading import load_data
from misc import metrics, utils
from FastXML.fastxml.trainer import Trainer
from FastXML.fastxml.fastxml import Inferencer
from FastXML.fastxml.weights import propensity
from scipy.sparse import csr_matrix


class FastxmlExp:
    def __init__(self, cfg):
        self.cfg = cfg
        print(f"> Loading data...")
        self.dataloaders = load_data(self.cfg)
        self.taxonomy = self.cfg['taxonomy']
        # Padding index set at 0, freeze=False to enable training on embeddings
        self.embeddings = torch.nn.Embedding.from_pretrained(self.dataloaders['embeddings'], padding_idx=0, freeze=True)
        print(f"> Loading model...")
        self.model = Trainer(
            # Number of trees in the ensemble
            n_trees=50,
            max_leaf_size=200,
            max_labels_per_leaf=200,
            n_jobs=16,
            # Number of iterations (max_iter in sklearn)
            n_epochs=10,
            # ['log', 'hinge']
            loss='hinge',
            # ['fastxml', 'dsimec']
            # I think that 'fastxml' put a 'l1' regularization while 'dsimec' a 'l2'
            optimization='dsimec',
            # False for FastXML and PFastXML, True for PFastreXML
            leaf_classifiers=True,
            # ['auto', 'sgd', 'liblinear']
            engine='liblinear',
            # C param for classifier (SVM or log regression)
            C=1,
            leaf_probs=True,
            verbose=False,
        )
        self.metrics_handler = {
            # Metrics during the training part
            # Evaluation on validation and test sets
            'eval_validation': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['validation_metrics']),
            'eval_test': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['test_metrics']),
        }


    def run(self):
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        split = 'train'
        training = True
        start_train = time.perf_counter()

        if training:
            # Get all data at the same time
            sparse_input = []
            all_labels = []
            for input_data, labels, _ in tqdm(self.dataloaders[f"global_{split}"], leave=False):
                for sample_tensor, sample_label in zip(input_data, labels):
                    # Do not take <PAD> tokens in the average
                    doc_embedding = torch.mean(self.embeddings(sample_tensor[sample_tensor != 0]), dim=0)
                    sparse_input.append(csr_matrix(doc_embedding.numpy()).astype(np.float32))
                    # Remove the taxonomy root
                    sample_label.remove(0)
                    all_labels.append(sample_label)

            # Train the model
            weights = propensity(all_labels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(sparse_input, all_labels, weights)

            self.model.save(str(self.cfg['paths']['model'].with_name('model')))

        stop_train = time.perf_counter()
        print(f"> End training after {utils.format_time_diff(start_train, stop_train)}")


        # 2. Evaluate the model
        inferencer = Inferencer(str(self.cfg['paths']['model'].with_name('model')), leaf_probs=True)
        splits_to_eval = ['validation', 'test']

        for split in splits_to_eval:
            print(f"\n> Evaluating the best model on the {split} set...")
            # Save predictions only on the test set
            self.save_pred = split == 'test'

            # 2.1 Global evaluation
            # Construct sparse matrices and the one-hot labels
            sparse_input = []
            all_complete_labels = []
            all_relevant_labels = None
            for batched_input, batched_labels, relevant_classes in tqdm(self.dataloaders[f"global_{split}"], leave=False):
                all_relevant_labels = relevant_classes
                for sample_tensor, sample_label in zip(batched_input, batched_labels):
                    # Do not take <PAD> tokens in the average
                    doc_embedding = torch.mean(self.embeddings(sample_tensor[sample_tensor != 0]), dim=0)
                    sparse_input.append(csr_matrix(doc_embedding.numpy()).astype(np.float32))
                    # Transform the label to one-hot encoding
                    one_hot_label = torch.zeros(self.taxonomy.n_nodes, dtype=torch.float32)
                    one_hot_label.scatter_(0, torch.tensor(sample_label), 1.)
                    all_complete_labels.append(one_hot_label)
            all_complete_labels = torch.stack(all_complete_labels, dim=0)

            # Get the predictions
            predictions_dict = inferencer.predict(sparse_input, fmt='dict')
            all_predictions = []
            for sample_dict in predictions_dict:
                pred = torch.tensor(list(sample_dict.keys()), dtype=torch.int64)
                score = torch.tensor(list(sample_dict.values()))
                assert len(pred) == len(score)
                pred_sample = torch.zeros(self.taxonomy.n_nodes, dtype=torch.float32)
                pred_sample.scatter_(dim=0, index=pred, src=score)
                all_predictions.append(pred_sample)
            all_predictions = torch.stack(all_predictions, dim=0)

            # Save predictions and labels
            if split == 'test':
                torch.save(all_predictions, self.cfg['paths'][f"{split}_predictions"])
                torch.save(all_complete_labels.bool(), self.cfg['paths'][f"{split}_labels"])
                torch.save(all_relevant_labels, self.cfg['paths'][f"{split}_relevant_labels"])

            # Filter predictions and labels and compute metrics
            filtered_predictions = all_predictions[:, all_relevant_labels]
            filtered_labels = all_complete_labels[:, all_relevant_labels]
            global_metrics, _n_docs = metrics.get_xml_metrics(
                filtered_predictions,
                filtered_labels,
                self.cfg['k_list'],
                self.cfg['loss_function'],
                self.cfg['threshold'],
            )

            # Gather metrics for the global evaluation
            print(f"> Results on the {split} set:")
            for metric_name, metric_value in global_metrics.items():
                new_row_dict = {
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'metric': metric_name,
                    'value': metric_value,
                }
                self.metrics_handler[f"eval_{split}"].add_row(new_row_dict)
                print(f">> {metric_name} -> {metric_value}")

            # Level metrics
            level_metrics_handler = utils.MetricsHandler(columns=['model', 'level', 'metric', 'value'], output_path=self.cfg['paths'][f"{split}_level_metrics"])
            level_metrics = metrics.get_metrics_per_level(
                all_predictions,
                all_complete_labels,
                self.cfg['k_list'],
                self.taxonomy,
                self.cfg['loss_function'],
                self.cfg['threshold'],
            )
            # Aggregate level metrics
            for level, metrics_dict in level_metrics.items():
                print(f"\n>> Metrics at level {level}")
                for metric_name, metric_value in metrics_dict.items():
                    new_row_dict = {
                        'model': self.cfg['method'],
                        'level': level,
                        'metric': metric_name,
                        'value': metric_value,
                    }
                    level_metrics_handler.add_row(new_row_dict)
                    print(f">>> {metric_name} -> {metric_value}")


            # 2.2 Tasks evaluation
            for task_id, (input_data, labels, task_relevant_classes) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
                # Construct sparse matrices and the one-hot labels
                sparse_input = []
                task_labels = []
                for sample_tensor, sample_label in zip(input_data, labels):
                    # Do not take <PAD> tokens in the average
                    doc_embedding = torch.mean(self.embeddings(sample_tensor[sample_tensor != 0]), dim=0)
                    sparse_input.append(csr_matrix(doc_embedding.numpy()).astype(np.float32))
                    # Transform the label to one-hot encoding
                    one_hot_label = torch.zeros(self.taxonomy.n_nodes, dtype=torch.float32)
                    one_hot_label.scatter_(0, torch.tensor(sample_label), 1.)
                    task_labels.append(one_hot_label)
                task_labels = torch.stack(task_labels, dim=0)

                # Get the predictions
                predictions_dict = inferencer.predict(sparse_input, fmt='dict')
                task_predictions = []
                for sample_dict in predictions_dict:
                    pred = torch.tensor(list(sample_dict.keys()), dtype=torch.int64)
                    score = torch.tensor(list(sample_dict.values()))
                    assert len(pred) == len(score)
                    pred_sample = torch.zeros(self.taxonomy.n_nodes, dtype=torch.float32)
                    pred_sample.scatter_(dim=0, index=pred, src=score)
                    task_predictions.append(pred_sample)
                task_predictions = torch.stack(task_predictions, dim=0)

                # Filter predictions and labels and compute metrics
                filtered_predictions = task_predictions[:, task_relevant_classes]
                filtered_labels = task_labels[:, task_relevant_classes]
                task_metrics, _n_docs = metrics.get_xml_metrics(
                    filtered_predictions,
                    filtered_labels,
                    self.cfg['k_list'],
                    self.cfg['loss_function'],
                    self.cfg['threshold'],
                )

                # Average over documents and save metrics
                for metric_name, metric_value in task_metrics.items():
                    new_row_dict = {
                        'model': self.cfg['method'],
                        'task': task_id,
                        'metric': metric_name,
                        'value': metric_value,
                    }
                    self.metrics_handler[f"eval_{split}"].add_row(new_row_dict)

        utils.print_config(self.cfg)
