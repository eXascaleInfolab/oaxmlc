from tqdm import tqdm
import torch
import numpy as np
import time

from datahandler.dataloading import load_data
from misc import metrics, utils


class BaseExp:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.cuda.set_device(self.cfg['device'])
        torch.cuda.device(self.cfg['device'])
        print(f"> Loading data...")
        self.dataloaders = load_data(self.cfg)
        self.taxonomy = self.cfg['taxonomy']
        self.early_stopping = utils.EarlyStopping(max_patience=self.cfg['patience'], cfg=self.cfg)
        self.metrics_handler = {
            # Metrics during the training part
            'training': utils.MetricsHandler(columns=['epoch', 'split', 'model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['metrics']),
            # Evaluation on validation and test sets
            'eval_validation': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['validation_metrics']),
            'eval_test': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['test_metrics']),
        }


    # These methods will be overridden in the children classes
    def load_model(self):
        print(f">> load_model method should be overridden in the child class")
        import sys; sys.exit()

    def inference_eval(self, input_data, **kwargs):
        print(f">> inference_eval method should be overridden in the child class")
        import sys; sys.exit()

    def optimization_loop(self, input_data, labels, **kwargs):
        print(f">> optimization_loop method should be overridden in the child class")
        import sys; sys.exit()

    def run_init(self):
        print(f">> run_init method should be overridden in the child class")
        import sys; sys.exit()


    # To save memory, do no compute the gradients since we do not need them here
    @torch.no_grad()
    def eval_step(self, split, epoch, metrics_handler, verbose=False, save_pred=False, final_evaluation=False):
        # Return all metrics for all tasks and the global evaluation
        returned_dict = {}

        # 1. Global evaluation
        # Storages for global predictions
        all_predictions = []
        all_complete_labels = []
        all_relevant_labels = None

        for input_data, labels, column_indices in tqdm(self.dataloaders[f"global_{split}"], leave=False):
            # Send to device
            input_data = input_data.to(self.cfg['device'])
            predictions = self.inference_eval(input_data)
            # Add predictions and labels to the storage
            all_predictions.append(predictions.cpu())
            all_complete_labels.append(labels)
            # Single tensor for all documents
            all_relevant_labels = column_indices[0]

        # Stack in one tensor
        all_predictions = torch.concat(all_predictions, dim=0)
        all_complete_labels = torch.concat(all_complete_labels, dim=0)

        # Save predictions, labels and relevant labels
        if save_pred:
            torch.save(all_predictions, self.cfg['paths'][f"{split}_predictions"])
            torch.save(all_complete_labels.bool(), self.cfg['paths'][f"{split}_labels"])
            # relevant labels are the same in the global evaluation
            torch.save(all_relevant_labels, self.cfg['paths'][f"{split}_relevant_labels"])

        # Mask predictions, labels and compute metrics
        filtered_predictions = all_predictions[:, all_relevant_labels]
        filtered_labels = all_complete_labels[:, all_relevant_labels]
        metrics_global, _n_docs = metrics.get_xml_metrics(
            filtered_predictions,
            filtered_labels,
            self.cfg['k_list'] if final_evaluation else self.cfg['k_list_eval_perf'],
            self.cfg['loss_function'],
            self.cfg['threshold'],
        )

        # Gather metrics for the global evaluation
        if verbose: print(f"> Results on the {split} set:")
        for metric_name, metric_value in metrics_global.items():
            if metrics_handler == 'training':
                new_row_dict = {
                    'epoch': epoch,
                    'split': split,
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'metric': metric_name,
                    'value': metric_value,
                }
            else:
                new_row_dict = {
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'metric': metric_name,
                    'value': metric_value,
                }
            self.metrics_handler[metrics_handler].add_row(new_row_dict)
            if verbose: print(f">> {metric_name} -> {metric_value}")
        # Add the computed metrics to the returned storage
        returned_dict[self.cfg['all_tasks_key']] = metrics_global

        # Compute the metrics per level at the final evaluation
        if final_evaluation:
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


        # 2. Tasks evaluation
        for task_id, (batched_input, batched_labels, column_indices) in enumerate(tqdm(self.dataloaders[f"tasks_{split}"], leave=False)):
            # Storages for this task
            task_predictions = []
            task_complete_labels = []

            for input_data, labels in zip(batched_input, batched_labels):
                # Predictions
                input_data = input_data.to(self.cfg['device'])
                predictions = self.inference_eval(input_data)

                # Add predictions and labels to the storage
                task_predictions.append(predictions.cpu())
                task_complete_labels.append(labels)

            # Stack all predictions and labels in one tensor
            task_predictions = torch.concat(task_predictions, dim=0)
            task_complete_labels = torch.concat(task_complete_labels, dim=0)

            # Mask predictions, label and compute metrics
            task_predictions = task_predictions[:, column_indices]
            task_complete_labels = task_complete_labels[:, column_indices]
            metrics_task, n_docs = metrics.get_xml_metrics(
                task_predictions,
                task_complete_labels,
                self.cfg['k_list'] if final_evaluation else self.cfg['k_list_eval_perf'],
                self.cfg['loss_function'],
                self.cfg['threshold'],
            )

            # Gather metrics for the task
            for metric_name, metric_value in metrics_task.items():
                if metrics_handler == 'training':
                    new_row_dict = {
                        'epoch': epoch,
                        'split': split,
                        'model': self.cfg['method'],
                        'task': task_id,
                        'metric': metric_name,
                        'value': metric_value,
                    }
                else:
                    new_row_dict = {
                        'model': self.cfg['method'],
                        'task': task_id,
                        'metric': metric_name,
                        'value': metric_value,
                    }
                self.metrics_handler[metrics_handler].add_row(new_row_dict)
            # Save dict for this task
            returned_dict[task_id] = metrics_task

        return returned_dict


    def run(self):
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        training = True
        epoch = 0
        split = 'train'
        start_train = time.perf_counter()
        # Misc initialization for the children classes
        self.run_init()

        # Evaluate the performance on the validation set before training
        before_training = self.eval_step(split='validation', epoch=epoch, metrics_handler='training')
        print(f">> Epoch {epoch} | Validation loss -> {before_training[self.cfg['all_tasks_key']]['loss']} | prec@1 -> {before_training[self.cfg['all_tasks_key']]['precision@1']}")

        while training:
            epoch += 1
            # Storages for stats
            train_losses = []
            n_docs_per_batch = []
            grad_norms = []
            grad_lengths = []
            # Model in training mode, e.g. activate dropout if specified
            for input_data, labels, _ in tqdm(self.dataloaders[f"global_{split}"], leave=False):
                # Send everything on device
                input_data = input_data.to(self.cfg['device'])
                labels = labels.to(self.cfg['device'])
                # Optimization loop
                train_loss, gradients_and_lengths = self.optimization_loop(input_data, labels)
                # Aggregate train loss and gradient norms
                train_losses.append(train_loss)
                n_docs_per_batch.append(len(input_data))
                grad_norms.append(gradients_and_lengths[0])
                grad_lengths.append(gradients_and_lengths[1])

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
            print(f">> Epoch {epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_metrics[self.cfg['all_tasks_key']]['loss']:.3f} | Gradients norm -> {np.average(grad_norms, weights=grad_lengths):.6f}")

            # Check if we stop training or not
            training = self.early_stopping.checkpoint(self.model, val_metrics[self.cfg['all_tasks_key']], epoch)

        stop_train = time.perf_counter()
        print(f"> End training at epoch {epoch} after {utils.format_time_diff(start_train, stop_train)}")

        # Evaluate the best model
        splits_to_eval = ['validation', 'test']
        # Reload best model
        self.load_model()
        for split in splits_to_eval:
            print(f"\n> Evaluating the best model on the {split} set...")
            self.eval_step(split=split, epoch='best', verbose=True, metrics_handler=f"eval_{split}", save_pred=split=='test', final_evaluation=True)

        utils.print_config(self.cfg)
