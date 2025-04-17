import torch

from models.model_loader import ModelLoader
from algorithms.base_exp import BaseExp


class XmlcnnExp(BaseExp):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(self.cfg)
        print(f"> Initialize model...")
        self.model = ModelLoader.get_model(self.cfg, embeddings=self.dataloaders['embeddings']).to(self.cfg['device'])
        self.model.init_plaincls(self.cfg['taxonomy'].n_nodes)
        self.optimizer = self.cfg['optimizer'](self.model.parameters(), lr=self.cfg['learning_rate'])

    def load_model(self):
        with open(self.cfg['paths']['state_dict'], "rb") as f:
            self.model.load_state_dict(torch.load(f, weights_only=True, map_location=self.cfg['device']))
        self.optimizer = self.cfg['optimizer'](self.model.parameters(), lr=self.cfg['learning_rate'])

    def inference_eval(self, input_data):
        predictions = self.model(input_data)
        return predictions

    def optimization_loop(self, input_data, labels):
        # Compute predictions and loss
        predictions = self.model(input_data)
        loss = self.cfg['loss_function'](predictions, labels)
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Compute gradient norms
        gradient_norms = []
        gradient_lengths = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradient_norms.append(torch.norm(param.grad).cpu())
                gradient_lengths.append(len(param.grad))
        self.optimizer.step()
        return loss.cpu().item(), (gradient_norms, gradient_lengths)

    def run_init(self):
        pass
