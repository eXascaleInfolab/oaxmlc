import time
import shutil
import torch
from brutelogger import BruteLogger

from misc.utils import print_time, format_time_diff

if __name__ == '__main__':
    print("You should not call this directly, see inside the `configs` folder.")
    import sys; sys.exit()


# Set max number of threads used for one experiment
n_threads = 4
torch.set_num_threads(min(torch.get_num_threads(), n_threads))
torch.set_num_interop_threads(min(torch.get_num_interop_threads(), n_threads))

# Select the correct model/architecture according the method/algorithm picked
model_name = {
    'match': 'match',
    'xmlcnn': 'xmlcnn',
    'attentionxml': 'attentionxml',
    'hector': 'hector',
    'tamlec': 'hector',
    'fastxml': 'fastxml',
    'lightxml': 'match',
    'cascadexml':'match',
    'parabel': 'parabel',
}

# Batch size for hector and tamlec
batch_size = {
    'hector': {
        'oatopics': 64,
        'oaconcepts': 20,
    },
    'tamlec': {
        'oatopics': 64,
        'oaconcepts': 40,
    },
}

# Number of accumulations for hector and tamlec
accum_iter = {
    'hector': {
        'oatopics': 5,
        'oaconcepts': 2,
    },
    'tamlec': {
        'oatopics': 5,
        'oaconcepts': 2,
    },
}

# Epochs for the patience for each method
patience = {
    'match': 5,
    'xmlcnn': 5,
    'attentionxml': 5,
    'hector': 3,
    'tamlec': 3,
    'fastxml': None,
    'lightxml': 5,
    'cascadexml': 5,
    'parabel': None,
}

# Specific threshold for the positives/negatives in the metrics
thresholds = {
    'oatopics': {
        'tamlec': 0.309,
        'hector': 0.179,
        'attentionxml': 0.342,
        'xmlcnn': 0.309,
        'match': 0.328,
        'lightxml': 0.314,
        'fastxml': 0.316,
        'cascadexml': 0.305,
        'parabel': 0.336,
    },
    'oaconcepts': {
        'tamlec': 0.424,
        'hector': 0.061,
        'attentionxml': 0.36,
        'xmlcnn': 0.316,
        'match': 0.361,
        'lightxml': 0.296,
        'fastxml': 0.348,
        'cascadexml': 0.52,
        'parabel': 0.5,
    },
}


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        assert self.cfg['method'] in ['match', 'xmlcnn', 'attentionxml', 'hector', 'tamlec', 'fastxml', 'lightxml', 'cascadexml', 'parabel'], f"{self.cfg['method']} not available, choose in ['match', 'xmlcnn', 'attentionxml', 'hector', 'tamlec', 'fastxml', 'lightxml', 'cascadexml', 'parabel']"

        # Create all paths
        # Output path set for the experiment
        output_path = self.cfg['output_path'] / self.cfg['exp_name'].stem
        predictions_path = output_path / 'predictions'
        preprocessed_data_path = self.cfg['dataset_path'] / 'preprocessed_data'
        self.cfg['paths'] = {
            'output': output_path,
            'dataset': self.cfg['dataset_path'],
            # Metrics
            'metrics': output_path / 'metrics.csv',
            'validation_metrics': output_path / f"{self.cfg['method']}_validation_metrics.csv",
            'test_metrics': output_path / f"{self.cfg['method']}_test_metrics.csv",
            'validation_level_metrics': output_path / f"{self.cfg['method']}_validation_level_metrics.csv",
            'test_level_metrics': output_path / f"{self.cfg['method']}_test_level_metrics.csv",
            # Saved models
            'model': output_path / 'model.pt',
            'state_dict': output_path / 'model_state_dict.pt',
            # Saved predictions
            'predictions_folder': predictions_path,
            'test_predictions': predictions_path / 'test_predictions.pt',
            'test_labels': predictions_path / 'test_labels.pt',
            'test_relevant_labels': predictions_path / 'relevant_labels.pt',
            # Saved pre-processed data
            'preprocessed_data': preprocessed_data_path,
            # Data for hector
            'taxos_hector': preprocessed_data_path / 'taxos_hector.pt',
            # Data for tamlec
            'taxos_tamlec': preprocessed_data_path / 'taxos_tamlec.pt',
            # Miscellaneous data
            'taxonomy': preprocessed_data_path / 'taxonomy.pt',
            'embeddings': preprocessed_data_path / 'embeddings.pt',
            'task_to_subroot': preprocessed_data_path / 'task_to_subroot.pt',
            'label_to_tasks': preprocessed_data_path / 'label_to_tasks.pt',
            'src_vocab': preprocessed_data_path / 'src_vocab.pt',
            'trg_vocab': preprocessed_data_path / 'trg_vocab.pt',
            'abstract_dict': preprocessed_data_path / 'abstract_dict.pt',
            'tasks_size': preprocessed_data_path / 'tasks_size.pt',
            'data': preprocessed_data_path / 'documents',
            'global_datasets': preprocessed_data_path / 'global_datasets.pt',
            'tasks_datasets': preprocessed_data_path / 'tasks_datasets.pt',
            'tokenizer': preprocessed_data_path / 'tokenizer.model',
            'vocabulary': preprocessed_data_path / 'tokenizer.vocab',
            'dataset_stats': preprocessed_data_path / 'dataset_stats',
            'drawn_tasks': preprocessed_data_path / 'dataset_stats' / 'drawn_tasks',
        }
        # Delete paths outside the newly-created dictionary
        del self.cfg['output_path']
        del self.cfg['dataset_path']

        # Create the folders
        self.cfg['paths']['output'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['predictions_folder'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['preprocessed_data'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['data'].mkdir(exist_ok=True, parents=True)
        self.cfg['paths']['drawn_tasks'].mkdir(exist_ok=True, parents=True)
        # Save everything that is printed on the console to a log file
        BruteLogger.save_stdout_to_file(path=self.cfg['paths']['output'], fname="all_console.log")
        # Copy the config file in the output_directory
        shutil.copy2(self.cfg['exp_name'], self.cfg['paths']['output'])

        # Size of embedding space
        self.cfg['emb_dim'] = 300
        # Key used in metrics file to represent average of tasks
        self.cfg['all_tasks_key'] = 'global'
        self.cfg['model_name'] = model_name[self.cfg['method']]
        self.cfg['dataset'] = self.cfg['paths']['dataset'].name
        try:
            self.cfg['threshold'] = thresholds[self.cfg['dataset']][self.cfg['method']]
        except KeyError:
            self.cfg['threshold'] = 0.5

        # Training loss function, optimizer and batch sizes
        self.cfg['loss_function'] = torch.nn.BCELoss()
        self.cfg['optimizer'] = torch.optim.AdamW
        if self.cfg['method'] in ['hector', 'tamlec']:
            self.cfg['batch_size_train'] = batch_size[self.cfg['method']][self.cfg['dataset']]
            self.cfg['batch_size_eval'] = batch_size[self.cfg['method']][self.cfg['dataset']]
            self.cfg['tamlec_params']['accum_iter'] = accum_iter[self.cfg['method']][self.cfg['dataset']]
        else:
            self.cfg['batch_size_train'] = 64
            self.cfg['batch_size_eval'] = 256
        self.cfg['patience'] = patience[self.cfg['method']]

        # For hector setup default parameters
        if self.cfg['method'] == 'hector':
            self.cfg['tamlec_params']['width_adaptive'] = False
            self.cfg['tamlec_params']['decoder_adaptative'] = 0
            self.cfg['tamlec_params']['tasks_size'] = False
            self.cfg['tamlec_params']['width_adaptive'] = False
            self.cfg['tamlec_params']['freeze'] = False
            self.cfg['tamlec_params']['with_bias'] = False

        assert self.cfg['tokenization_mode'] in ['word', 'bpe', 'unigram'], f"{self.cfg['tokenization_mode']} should be ['word', 'bpe', 'unigram']"

    def main_run(self):
        start_time = time.perf_counter()

        if self.cfg['method'] == 'match':
            from algorithms.match import MatchExp
            print_time(f"Starting MATCH experiment")
            experiment = MatchExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'xmlcnn':
            from algorithms.xmlcnn import XmlcnnExp
            print_time(f"Starting XMLCNN experiment")
            experiment = XmlcnnExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'attentionxml':
            from algorithms.attentionxml import AttentionxmlExp
            print_time(f"Starting AttentionXML experiment")
            experiment = AttentionxmlExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'hector':
            from algorithms.hector import HectorExp
            print_time(f"Starting HECTOR experiment")
            experiment = HectorExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'tamlec':
            from algorithms.tamlec import TamlecExp
            print_time(f"Starting TAMLEC experiment")
            experiment = TamlecExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'fastxml':
            from algorithms.fastxml import FastxmlExp
            print_time(f"Starting FASTXML experiment")
            experiment = FastxmlExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'lightxml':
            from algorithms.lightxml import LightxmlExp
            print_time(f"Starting LightXML experiment")
            experiment = LightxmlExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'cascadexml':
            from algorithms.cascadexml import CascadexmlExp
            print_time(f"Starting CascadeXML experiment")
            experiment = CascadexmlExp(self.cfg)
            experiment.run()
        elif self.cfg['method'] == 'parabel':
            from algorithms.parabel import ParabelExp
            print_time(f"Starting Parabel experiment")
            experiment = ParabelExp(self.cfg)
            experiment.run()

        stop_time = time.perf_counter()
        print_time(f"Experiment ended in {format_time_diff(start_time, stop_time)}")
