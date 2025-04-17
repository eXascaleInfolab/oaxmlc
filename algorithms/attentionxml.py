import torch
import numpy as np
from ruamel.yaml import YAML
from pathlib import Path
from AttentionXML.deepxml.networks import AttentionRNN
from AttentionXML.deepxml.models import Model

from algorithms.base_exp import BaseExp


class AttentionxmlExp(BaseExp):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(self.cfg)
        print(f"> Initialize model...")
        yaml = YAML(typ='safe')
        model_cnf = yaml.load(Path("AttentionXML/configure/models/baseconfig.yaml"))
        self.model = Model(
            network=AttentionRNN,
            labels_num=self.taxonomy.n_nodes,
            model_path=f"{self.cfg['paths']['output']}/model_state_dict_attention.pt",
            emb_init=np.array(self.dataloaders['embeddings']),
            emb_size=self.cfg['emb_dim'],
            **model_cnf['model']
        )
        self.model.get_optimizer(lr=self.cfg['learning_rate'])
        self.sigmoid = torch.nn.Sigmoid()

    def load_model(self):
        with open(self.cfg['paths']['model'], "rb") as f:
            self.model = torch.load(f, map_location=self.cfg['device'])
        self.model.get_optimizer(lr=self.cfg['learning_rate'])

    def inference_eval(self, input_data):
        scores = self.model.model(input_data)
        predictions = self.sigmoid(scores)
        return predictions

    def optimization_loop(self, input_data, labels):
        train_loss, gradients_and_lengths = self.model.train_step(input_data, labels)
        return train_loss, gradients_and_lengths

    def run_init(self):
        pass
