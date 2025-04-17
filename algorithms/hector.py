import torch
import numpy as np
import nltk
from tqdm import tqdm
import time

from Hector.hector import Hector
from Hector.prediction_handler import CustomXMLHolder
from datahandler.dataloading import load_data
from misc import metrics, utils


class HectorExp:
    def __init__(self, cfg):
        # Download from nltk
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')

        self.cfg = cfg
        torch.cuda.set_device(self.cfg['device'])
        torch.cuda.device(self.cfg['device'])
        print(f"> Loading data...")
        self.dataloaders = load_data(self.cfg)
        print(f"> Loading model...")
        self.model = Hector(
            src_vocab=self.cfg['tamlec_params']['src_vocab'],
            tgt_vocab=self.cfg['tamlec_params']['trg_vocab'],
            path_to_glove=".vector_cache/glove.840B.300d.gensim",
            abstract_dict=self.cfg['tamlec_params']['abstract_dict'],
            taxonomies=self.cfg['tamlec_params']['taxos_hector'],
            width_adaptive=self.cfg['tamlec_params']['width_adaptive'],
            decoder_adaptative=self.cfg['tamlec_params']['decoder_adaptative'],
            tasks_size=self.cfg['tamlec_params']['tasks_size'],
            gpu_target=self.cfg['device'],
            with_bias=self.cfg['tamlec_params']['with_bias'],
            Number_src_blocs=6,
            Number_tgt_blocs=6,
            dim_src_embedding=300,
            dim_tgt_embedding=600,
            dim_feed_forward=2048,
            number_of_heads=12,
            dropout=0.1,
            learning_rate=self.cfg['learning_rate'],
            beta1=0.9,
            beta2=0.99,
            epsilon=1e-8,
            weight_decay=0.01,
            gamma=.99998,
            accum_iter=self.cfg['tamlec_params']['accum_iter'],
            # 0.0 < x <= 0.1
            loss_smoothing=self.cfg['tamlec_params']['loss_smoothing'],
            max_padding_document=self.cfg['seq_length'],
            max_number_of_labels=20,
        )
        self.early_stopping = utils.EarlyStopping(max_patience=self.cfg['patience'], cfg=self.cfg)
        self.metrics_handler = {
            # Metrics during the training part
            'training': utils.MetricsHandler(columns=['epoch', 'split', 'model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['metrics']),
            # Evaluation on validation and test sets
            'eval_validation': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['validation_metrics']),
            'eval_test': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['test_metrics']),
        }


    # To save memory, do no compute the gradients since we do not need them here
    @torch.no_grad()
    def eval_step(self, split, epoch, metrics_handler):
        # Set model to eval() modifies the behavior of certain layers e.g. dropout, normalization layers
        self.model.eval()
        # Return all metrics for all tasks and the global evaluation
        returned_dict = {}

        # 1. Global evaluation
        n_docs_per_batch = []
        losses = []
        all_precisions_at = {k: [] for k in self.cfg['k_list_eval_perf']}
        total_masses = []
        for input_data, labels, _ in tqdm(self.dataloaders[f"global_{split}"], leave=False):
            # For hector always set task_id=0 as the whole taxonomy is taken into account
            loss, precisions, total_mass = self.model.eval_batch(documents_tokens=input_data, labels_tokens=labels, task_id=0)
            losses.append(loss.cpu().item())
            for k in self.cfg['k_list_eval_perf']:
                all_precisions_at[k].append(precisions[k])
            total_masses.append(total_mass)
            n_docs_per_batch.append(len(input_data))

        # Gather global metrics
        metrics_global = {
            'loss': np.average(losses, weights=n_docs_per_batch),
            'mass': np.average(total_masses, weights=n_docs_per_batch),
        }
        for k in self.cfg['k_list_eval_perf']: metrics_global[f"precision@{k}"] = np.average(all_precisions_at[k], weights=n_docs_per_batch)
        returned_dict[self.cfg['all_tasks_key']] = metrics_global

        # Save metrics
        for metric_name, metric_value in metrics_global.items():
            new_row_dict = {
                'epoch': epoch,
                'split': split,
                'model': self.cfg['method'],
                'task': self.cfg['all_tasks_key'],
                'metric': metric_name,
                'value': metric_value,
            }
            self.metrics_handler[metrics_handler].add_row(new_row_dict)
        # Add the computed metrics to the returned storage
        returned_dict[self.cfg['all_tasks_key']] = metrics_global


        # 2. Tasks evaluation
        for task_id, (batched_input, batched_labels, _) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
            n_docs_per_task = []
            losses = []
            all_precisions_at = {k: [] for k in self.cfg['k_list_eval_perf']}
            total_masses = []
            for input_data, labels in zip(batched_input, batched_labels):
                # For hector always set task_id=0 as the whole taxonomy is taken into account
                loss, precisions, total_mass = self.model.eval_batch(documents_tokens=input_data, labels_tokens=labels, task_id=0)
                losses.append(loss.cpu().item())
                for k in self.cfg['k_list_eval_perf']:
                    all_precisions_at[k].append(precisions[k])
                total_masses.append(total_mass)
                n_docs_per_task.append(len(input_data))

            # Gather metrics for the task
            metrics_dict = {
                'loss': np.average(losses, weights=n_docs_per_task),
                'mass': np.average(total_masses, weights=n_docs_per_task),
            }
            for k in self.cfg['k_list_eval_perf']: metrics_dict[f"precision@{k}"] = np.average(all_precisions_at[k], weights=n_docs_per_task)

            # Save metrics
            for metric_name, metric_value in metrics_dict.items():
                new_row_dict = {
                    'epoch': epoch,
                    'split': split,
                    'model': self.cfg['method'],
                    'task': task_id,
                    'metric': metric_name,
                    'value': metric_value,
                }
                self.metrics_handler[metrics_handler].add_row(new_row_dict)
            # Add the computed metrics to the returned storage
            returned_dict[task_id] = metrics_dict

        return returned_dict


    def run(self):
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        training = True
        epoch = 0
        split = 'train'
        start_train = time.perf_counter()

        # Evaluate the performance on the validation set before training
        before_training = self.eval_step(split='validation', epoch=epoch, metrics_handler='training')
        print(f">> Epoch {epoch} | Validation loss -> {before_training[self.cfg['all_tasks_key']]['loss']} | prec@1 -> {before_training[self.cfg['all_tasks_key']]['precision@1']}")

        while training:
            epoch += 1
            train_losses = []
            n_docs_per_batch = []
            # Model in training mode, e.g. activate dropout if specified
            self.model.train()
            for input_data, labels, _ in tqdm(self.dataloaders[f"global_{split}"], leave=False):
                # Train on batch
                # For hector always set task_id=0 as the whole taxonomy is taken into account
                loss = self.model.train_on_batch(documents_tokens=input_data, labels_tokens=labels, task_id=0)
                train_losses.append(loss.cpu().item())
                n_docs_per_batch.append(len(input_data))

            # Compute train loss for the whole epoch
            mean_train_loss = np.average(train_losses, weights=n_docs_per_batch)
            new_row_dict = {
                'epoch': epoch,
                'split': split,
                'model': self.cfg['method'],
                'task': self.cfg['all_tasks_key'],
                'metric': 'loss',
                'value': mean_train_loss,
            }
            self.metrics_handler['training'].add_row(new_row_dict)

            # Evaluate the performance on the validation set
            val_metrics = self.eval_step(split='validation', epoch=epoch, metrics_handler='training')
            print(f">> Epoch {epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_metrics[self.cfg['all_tasks_key']]['loss']:.3f}")

            # Check if we stop training or not
            training = self.early_stopping.checkpoint(self.model, val_metrics[self.cfg['all_tasks_key']], epoch)

        stop_train = time.perf_counter()
        print(f"> End training at epoch {epoch} after {utils.format_time_diff(start_train, stop_train)}")

        # Evaluate the best model
        splits_to_eval = ['validation', 'test']
        # Reload best model
        with open(self.cfg['paths']['model'], "rb") as f:
            self.model = torch.load(f, map_location=self.cfg['device'])
        for split in splits_to_eval:
            print(f"\n> Evaluating the best model on the {split} set...")
            self.final_evaluation(split=split, metrics_handler=f"eval_{split}", save_pred=split=='test')

        utils.print_config(self.cfg)


    @torch.no_grad()
    def final_evaluation(self, split, metrics_handler, save_pred):
        # 1. Global evaluation
        # Aggregate all data
        input_data = []
        labels = []
        all_relevant_labels = None
        for batched_input, batched_labels, column_indices in self.dataloaders[f"global_{split}"]:
            input_data.append(torch.vstack(batched_input))
            for labels_in_batch in batched_labels:
                labels.append(labels_in_batch)
                all_relevant_labels = column_indices
        input_data = torch.vstack(input_data)

        # Set model to eval() modifies the behavior of certain layers e.g. dropout, normalization layers
        self.model.eval()
        # Compute predictions
        # For hector always set task_id=0 as the whole taxonomy is taken into account
        xml = CustomXMLHolder(text_batch=input_data, task_id=0, beam_parameter=10, hector=self.model, proba_operator="MAX_PROBA")
        predictions, scores = xml.run_predictions(batch_size=256)

        # Transform predictions and labels into one-hot encoding
        all_predictions = []
        all_complete_labels = []
        for pred, score, label in zip(tqdm(predictions), scores, labels):
            assert len(pred) == len(score)
            pred = torch.tensor(pred)
            score = torch.tensor(score)
            label = torch.tensor(label)
            pred_sample = torch.zeros(self.cfg['taxonomy'].n_nodes+1, dtype=torch.float64)
            pred_sample.scatter_(dim=0, index=pred, src=score)
            assert pred_sample[-1] == 0.
            all_predictions.append(pred_sample)
            label_sample = torch.zeros(self.cfg['taxonomy'].n_nodes+1, dtype=torch.float64)
            label_sample.scatter_(0, label, 1.)
            assert label_sample[-1] == 0.
            all_complete_labels.append(label_sample)

        # Aggregate predictions and labels
        all_predictions = torch.stack(all_predictions, dim=0)
        all_complete_labels = torch.stack(all_complete_labels, dim=0)

        # Save predictions and labels
        if save_pred:
            torch.save(all_relevant_labels, self.cfg['paths'][f"{split}_relevant_labels"])
            torch.save(all_predictions, self.cfg['paths'][f"{split}_predictions"])
            torch.save(all_complete_labels.bool(), self.cfg['paths'][f"{split}_labels"])

        # Filter predictions and labels
        filtered_predictions = all_predictions[:, all_relevant_labels]
        filtered_labels = all_complete_labels[:, all_relevant_labels]

        # Compute metrics
        metrics_global, _n_docs = metrics.get_xml_metrics(
            filtered_predictions,
            filtered_labels,
            self.cfg['k_list'],
            self.cfg['loss_function'],
            self.cfg['threshold'],
        )

        # Average over documents and save metrics
        print(f"> Results on the {split} set")
        for metric_name, metric_value in metrics_global.items():
            new_row_dict = {
                'model': self.cfg['method'],
                'task': self.cfg['all_tasks_key'],
                'metric': metric_name,
                'value': metric_value,
            }
            self.metrics_handler[f"eval_{split}"].add_row(new_row_dict)
            print(f">> {metric_name} -> {metric_value}")

        # Compute the metrics per level at the final evaluation
        level_metrics_handler = utils.MetricsHandler(columns=['model', 'level', 'metric', 'value'], output_path=self.cfg['paths'][f"{split}_level_metrics"])
        level_metrics = metrics.get_metrics_per_level(
            all_predictions,
            all_complete_labels,
            self.cfg['k_list'],
            self.cfg['taxonomy'],
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


        # 2. Tasks evaluation
        for task_id, (batched_input, batched_labels, task_relevant_classes) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
            # Aggregate batched data (3D) to un-batched data (2D)
            input_data = torch.vstack(batched_input)
            labels = []
            for labels_in_batch in batched_labels:
                for label_sample in labels_in_batch:
                    labels.append(label_sample)

            # Compute predictions
            # For hector always set task_id=0 as the whole taxonomy is taken into account
            xml = CustomXMLHolder(text_batch=input_data, task_id=0, beam_parameter=10, hector=self.model, proba_operator="MAX_PROBA")
            predictions, scores = xml.run_predictions(batch_size=256)

            # Transform predictions and labels into one-hot encoding
            all_predictions = []
            all_complete_labels = []
            for pred, score, label in zip(predictions, scores, labels):
                assert len(pred) == len(score)
                pred = torch.tensor(pred)
                score = torch.tensor(score)
                label = torch.tensor(label)
                pred_sample = torch.zeros(self.cfg['taxonomy'].n_nodes+1, dtype=torch.float64)
                pred_sample.scatter_(dim=0, index=pred, src=score)
                assert pred_sample[-1] == 0.
                all_predictions.append(pred_sample)
                label_sample = torch.zeros(self.cfg['taxonomy'].n_nodes+1, dtype=torch.float64)
                label_sample.scatter_(0, label, 1.)
                assert label_sample[-1] == 0.
                all_complete_labels.append(label_sample)

            # Aggregate and filter predictions and labels
            all_predictions = torch.stack(all_predictions, dim=0)
            all_predictions = all_predictions[:, task_relevant_classes]
            all_complete_labels = torch.stack(all_complete_labels, dim=0)
            all_complete_labels = all_complete_labels[:, task_relevant_classes]

            # Compute metrics
            metrics_task, _n_docs = metrics.get_xml_metrics(
                all_predictions,
                all_complete_labels,
                self.cfg['k_list'],
                self.cfg['loss_function'],
                self.cfg['threshold'],
            )

            # Average over documents and save for global aggregation
            for metric_name, metric_value in metrics_task.items():
                new_row_dict = {
                    'model': self.cfg['method'],
                    'task': task_id,
                    'metric': metric_name,
                    'value': metric_value,
                }
                self.metrics_handler[f"eval_{split}"].add_row(new_row_dict)
