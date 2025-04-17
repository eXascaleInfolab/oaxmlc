import torch

from models.model_loader import ModelLoader
from algorithms.base_exp import BaseExp


class MatchExp(BaseExp):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(self.cfg)
        print(f"> Initialize model...")
        self.model = ModelLoader.get_model(self.cfg, embeddings=self.dataloaders['embeddings']).to(self.cfg['device'])
        self.model.init_plaincls(self.taxonomy.n_nodes)
        self.optimizer = self.cfg['optimizer'](self.model.parameters(), lr=self.cfg['learning_rate'])
        self.relu = torch.nn.ReLU()
        self.lambda1 = 1e-8
        self.lambda2 = 1e-10

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

        # Output Regularization
        out_regs = torch.zeros(*predictions.size(), device=self.cfg['device'])
        out_regs[:, self.child_indices] = predictions[:, self.child_indices] - predictions[:, self.parent_indices]
        loss += self.lambda1 * torch.sum(self.relu(out_regs)).item()

        # Parameter Regularization
        weights = self.model.plaincls.out_mesh_dstrbtn.weight
        param_regs = torch.zeros(weights.size(1), predictions.size(1), device=self.cfg['device'])
        # Transpose to match the shapes
        param_regs[:, self.child_indices] = weights[self.parent_indices].T - weights[self.child_indices].T
        loss += self.lambda2 * 1/2 * torch.norm(param_regs, p=2) ** 2

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
        # Indices used for regularization terms
        # Get indices of children and their parents, excluding 'root' and '<pad>'
        self.child_indices = []
        self.parent_indices = []
        for child, idx in self.taxonomy.label_to_idx.items():
            if child not in ['root', '<pad>']:
                for parent in self.taxonomy.label_to_parents[child]:
                    self.child_indices.append(idx)
                    self.parent_indices.append(self.taxonomy.label_to_idx[parent])
        self.child_indices = torch.tensor(self.child_indices, device=self.cfg['device'])
        self.parent_indices = torch.tensor(self.parent_indices, device=self.cfg['device'])
