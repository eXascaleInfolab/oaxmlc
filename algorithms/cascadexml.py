import torch
import numpy as np
from sklearn.cluster import KMeans
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from models.cascadexml import CascadeXML
from models.model_loader import ModelLoader
from misc import utils
from algorithms.base_exp import BaseExp


class CascadexmlExp(BaseExp):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(self.cfg)
        print(f"> Get label embeddings...")
        self.backbone = ModelLoader.get_model(self.cfg, embeddings=self.dataloaders['embeddings']).to(self.cfg['device'])
        self.label_embeddings = self.labels_tfidf()

        print(f"> Creating clusters for {len(self.label_embeddings)} labels (root excluded)...")
        # Create two levels of clusters
        # Fine level, so just above all labels
        # Original paper makes a clustering with maximum size of 2 by cluster, so we do the same
        fine_cluster, fine_centers, self.fine_cluster_map = self.get_even_clusters(self.label_embeddings, cluster_size=2)
        fine_cluster = dict(sorted(fine_cluster.items(), key=lambda x: x[0]))
        print(f">> Got {len(fine_centers)} clusters at level 2")
        # Coarse level, first level
        coarse_cluster, coarse_centers, self.coarse_cluster_map = self.get_even_clusters(fine_centers, cluster_size=2)
        coarse_cluster = dict(sorted(coarse_cluster.items(), key=lambda x: x[0]))
        print(f">> Got {len(coarse_centers)} clusters at level 1")
        self.clusters = [list(coarse_cluster.values()), list(fine_cluster.values())]

        print(f"> Loading model...")
        self.model = CascadeXML(
            cfg=self.cfg,
            backbone=self.backbone,
            clusters=self.clusters,
        )
        self.model.to(self.cfg['device'])
        self.optimizer = self.cfg['optimizer'](self.model.parameters(), lr=self.cfg['learning_rate'])

        self.early_stopping = utils.EarlyStopping(max_patience=self.cfg['patience'], cfg=self.cfg)
        self.metrics_handler = {
            # Metrics during the training part
            'training': utils.MetricsHandler(columns=['epoch', 'split', 'model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['metrics']),
            # Evaluation on validation and test sets
            'eval_validation': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['validation_metrics']),
            'eval_test': utils.MetricsHandler(columns=['model', 'task', 'metric', 'value'], output_path=self.cfg['paths']['test_metrics']),
        }


    # From https://stackoverflow.com/questions/5452576/k-means-algorithm-variation-with-equal-cluster-size?rq=1
    def get_even_clusters(self, X, cluster_size):
        n_clusters = int(np.ceil(len(X)/cluster_size))
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, verbose=1, random_state=16)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
        distance_matrix = cdist(X, centers)
        clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
        clustering = {}
        cluster_map = {}
        for idx, cluster_idx in enumerate(clusters):
            try:
                clustering[cluster_idx].append(idx)
            except KeyError:
                clustering[cluster_idx] = [idx]
            cluster_map[idx] = cluster_idx
        centers = np.array([X[clusters == i].mean(axis=0) for i in range(n_clusters)])

        # Cluster map is a tensor so we can use vectorized lookup with other tensors
        cluster_map = dict(sorted(cluster_map.items(), key=lambda x: x[0]))
        cluster_map = torch.tensor(list(cluster_map.values()), dtype=torch.int64, device=self.cfg['device'])

        return clustering, centers, cluster_map


    def labels_tfidf(self):
        plain_texts = []
        # Root is skipped since it is never predicted
        # Get all abstracts from the labels
        for idx in range(1, self.taxonomy.n_nodes):
            label = self.taxonomy.idx_to_label[idx]
            abstract = self.taxonomy.label_to_abstract[label]
            abstract = abstract.lower().strip().replace('\n', '')
            # Remove all special characters, keep only letters, numbers and spaces
            abstract = re.sub(r"[^\w\s]", '', abstract)
            plain_texts.append(abstract)
        vectorizer = TfidfVectorizer()
        lab_embs = vectorizer.fit_transform(plain_texts)
        # Transform SciPy sparse matrix to Numpy array
        lab_embs = np.asarray(lab_embs.todense())

        return lab_embs


    def load_model(self):
        with open(self.cfg['paths']['state_dict'], "rb") as f:
            self.model.load_state_dict(torch.load(f, weights_only=True, map_location=self.cfg['device']))
        self.optimizer = self.cfg['optimizer'](self.model.parameters(), lr=self.cfg['learning_rate'])


    def inference_eval(self, input_data):
        all_probs, all_candidates, _ = self.model(input_data)
        predictions = []
        # -1 so we take predictions of the last level of the clustering, i.e. our class labels
        for pred, score in zip(all_candidates[-1], all_probs[-1]):
            assert len(pred) == len(score)
            # Highest index is a <PAD> class, this is because of the balanced clustering
            # and the way the code is written in the original model
            # Trick is to have one more class representing the <PAD>
            # And then make sure it is equal to 0
            pred_sample = torch.zeros(self.cfg['taxonomy'].n_nodes+1, dtype=torch.float32)
            # Shift back to our class indices, since we did not cluster the root node
            pred += 1
            pred_sample.scatter_(dim=0, index=pred.cpu(), src=score.cpu())
            # <PAD>  probability should be zero
            assert pred_sample[-1] == 0.
            # Also check that the root has a zero probability, so that the shift is working
            assert pred_sample[0] == 0.
            # Remove the padding probability
            predictions.append(pred_sample[:-1])
        predictions = torch.stack(predictions, dim=0)

        return predictions


    def optimization_loop(self, input_data, labels):
        # Labels for cascadexml are a list of lists of tensors
        # There is one list per level in the clustering (first list is for highest level, last list is all class labels)
        # Each list is of size batch_size
        coarse_level = []
        fine_level = []
        class_level = []
        for label in labels:
            class_labels = (label == 1.).nonzero(as_tuple=True)[0]
            # Remove the root
            class_labels = class_labels[class_labels != 0]
            class_labels -= 1
            class_level.append(class_labels)
            # Construct labels for the other levels in the clustering
            fine_labels = torch.unique(self.fine_cluster_map[class_labels])
            coarse_labels = torch.unique(self.coarse_cluster_map[fine_labels])
            fine_level.append(fine_labels)
            coarse_level.append(coarse_labels)
        cascadexml_labels = [coarse_level, fine_level, class_level]

        # Get predictions
        all_probs, all_candidates, loss = self.model(input_data, all_labels=cascadexml_labels)

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
