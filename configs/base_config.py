import sys
import os
# Set the cwd as outside the configs folder so that the imports work properly
sys.path.insert(0, os.getcwd())
from pathlib import Path

from misc.experiment import Experiment

cfg = {
    # Make sure to name the folder 'oaxmlc_topics' or 'oaxmlc_concepts'
    'dataset_path': Path("datasets/oaxmlc_topics"),
	'output_path': Path("output"),
	# Experiment name, do not change
    'exp_name': Path(__file__),
    'device': 'cuda:0',
	# ['match', 'xmlcnn', 'attentionxml', 'fastxml', 'hector', 'lightxml', 'cascadexml', 'parabel']
    'method': 'match',
    'learning_rate': 5e-5,
    # Length of the input sequences
    'seq_length': 128,
    # (Maximum) Vocabulary size
    'voc_size': 10000,
    # ['word', 'bpe', 'unigram']
    'tokenization_mode': 'word',
    # k to evaluate in the metrics for the final evaluation
    'k_list': list(range(1, 21)),
    # While training only, not final evaluation, to speed up metrics computation
    'k_list_eval_perf': [1,2,3,5],
    # Parameters specific to HECTOR
    'hector_params': {
        'loss_smoothing': 1e-2,
    },
}

if __name__ == '__main__':
    # DEBUG: uncaught exceptions drop you into ipdb for postmortem debugging
    import sys, IPython; sys.excepthook = IPython.core.ultratb.FormattedTB(mode="Context", call_pdb=True)
    exp = Experiment(cfg)
    exp.main_run()
