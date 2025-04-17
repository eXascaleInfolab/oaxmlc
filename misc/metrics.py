import torch
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import roc_curve


# Predictions should be the probabilities (i.e. between 0.0 and 1.0)
def get_xml_metrics(predictions, labels, k_list, loss_fn=None, threshold=0.5):
    metrics = {}
    n_docs = {}
    if loss_fn is not None:
        metrics['loss'] = loss_fn(predictions, labels).cpu().item()
        n_docs['loss'] = len(predictions)
    for k in k_list:
        # Take only documents that have at least k labels
        n_labels = torch.sum(labels, dim=1)
        filt_pred = predictions[n_labels >= k]
        filt_labels = labels[n_labels >= k]
        # If no documents remaining then skip this computation
        if len(filt_pred) == 0: continue
        precision, recall = precision_recall_at_k(filt_pred, filt_labels, k)
        metrics[f"precision@{k}"] = precision
        n_docs[f"precision@{k}"] = len(filt_pred)
        metrics[f"recall@{k}"] = recall
        n_docs[f"recall@{k}"] = len(filt_pred)
        metrics[f"f1score@{k}"] = hmean([precision, recall])
        n_docs[f"f1score@{k}"] = len(filt_pred)
        metrics[f"rankedsum@{k}"] = ranked_sum_at_k(filt_pred, filt_labels, k)
        n_docs[f"rankedsum@{k}"] = len(filt_pred)
        metrics[f"ndcg@{k}"] = ndcg_k_baseline(filt_pred, filt_labels, k)
        n_docs[f"ndcg@{k}"] = len(filt_pred)

    micro_precision, micro_recall, micro_f1score, macro_precision, macro_recall, macro_f1score = precision_recall(predictions, labels, threshold)
    metrics["micro_precision"] = micro_precision
    n_docs["micro_precision"] = len(predictions)
    metrics["micro_recall"] = micro_recall
    n_docs["micro_recall"] = len(predictions)
    metrics["micro_f1score"] = micro_f1score
    n_docs["micro_f1score"] = len(predictions)
    metrics["macro_precision"] = macro_precision
    n_docs["macro_precision"] = len(predictions)
    metrics["macro_recall"] = macro_recall
    n_docs["macro_recall"] = len(predictions)
    metrics["macro_f1score"] = macro_f1score
    n_docs["macro_f1score"] = len(predictions)

    return metrics, n_docs


# Predictions should be the probabilities (i.e. between 0.0 and 1.0)
def get_metrics_per_level(predictions, labels, k_list, taxonomy, loss_fn=None, threshold=0.5):
    metrics = {}
    for level, nodes in taxonomy.level_to_labels.items():
        # Skip level of the root, and level of the sub-roots
        if (level == 0) or (level == 1): continue
        # Filter predictions and labels for only nodes that appear at a given level
        label_indices = torch.tensor([taxonomy.label_to_idx[lab] for lab in nodes], device=predictions.device)
        level_metrics, n_docs = get_xml_metrics(predictions[:, label_indices], labels[:, label_indices], k_list, loss_fn, threshold)
        metrics[level] = level_metrics

    return metrics


# Precision at top k
# How many correct labels are there in the top k predictions
# Predictions should be the probabilities (i.e. between 0.0 and 1.0)
def precision_recall_at_k(predictions, labels, k):
    # Sort by highest score, and transform to index directly
    ranked_preds = torch.argsort(predictions, dim=1, descending=True, stable=True)
    # Get the top_k ranked labels per each sample
    top_k = ranked_preds[:, :k]
    top_k_labels = torch.gather(labels, dim=1, index=top_k)
    relevant_items = torch.sum(top_k_labels, dim=1)
    precisions = relevant_items / k
    recalls = relevant_items / torch.sum(labels, dim=1)

    return torch.mean(precisions, dim=0).cpu().item(), torch.mean(recalls, dim=0).cpu().item()


# Predictions should be the probabilities (i.e. between 0.0 and 1.0)
def ranked_sum_at_k(predictions, labels, k):
    # Sort by highest score, and transform to index directly
    ranked_preds = torch.argsort(predictions, dim=1, descending=True, stable=True)
    ranked_labels = torch.gather(labels, dim=1, index=ranked_preds)
    # Use cumsum to find the first occurrence of a relevant item
    cumulative_relevant = torch.cumsum(ranked_labels, dim=1)
    # Mask out all positions except for the first relevant one (where cumsum is k)
    first_relevant_mask = (cumulative_relevant == k)
    ranks = torch.arange(1.0, predictions.size(1) + 1.0).to(predictions.device)
    # Broadcast of ranks into first_relevant_mask
    first_relevant_ranks = ranks * first_relevant_mask
    # For each row, get the first non-zero rank (the rank of the first relevant item)
    # Transform 0.0 to 'inf' so we can use the min function
    first_relevant_ranks[first_relevant_ranks == 0.0] = float('inf')
    ranked_sums, _ = torch.min(first_relevant_ranks, dim=1)

    return torch.mean(ranked_sums, dim=0).cpu().item()


# Given probability predictions, get the best thresholds for the decision boundary
# i.e. at which threshold we should discriminate positive VS negative predictions
# Predictions should be the probabilities (i.e. between 0.0 and 1.0)
def get_thresholds(predictions, labels):
    flatten_preds = torch.flatten(predictions)
    flatten_labels = torch.flatten(labels)
    n_positives = torch.sum(flatten_labels == 1).item()
    n_negatives = torch.sum(flatten_labels == 0).item()
    flatten_preds = flatten_preds.numpy()
    flatten_labels = flatten_labels.numpy()
    assert n_positives + n_negatives == len(flatten_labels)
    assert len(flatten_preds) == len(flatten_labels)
    # recall <=> true positive rate
    fprs, recalls, thresholds = roc_curve(flatten_labels, flatten_preds, drop_intermediate=True)
    # First threshold is here so that the curve can start at fpr=0 and tpr=0 but we do not need this
    fprs = fprs[1:]
    recalls = recalls[1:]
    thresholds = thresholds[1:]
    recalls = np.nan_to_num(recalls, nan=0.)
    # Get number of TP and FP
    n_tp = recalls * n_positives
    n_fp = fprs * n_negatives
    # Compute the precision
    precisions = n_tp / (n_tp + n_fp)
    precisions = np.nan_to_num(precisions, nan=0.)
    # Finally compute the f1 score
    f1s = 2 * precisions * recalls / (precisions + recalls)
    f1s = np.nan_to_num(f1s, nan=0.)
    # Sort in descending order w.r.t. the f1 scores
    # Then we will take the threshold that maximized the f1 score
    sorted_thresholds = thresholds[np.argsort(f1s)[::-1]]
    print(f"Avg F1 -> {np.mean(np.sort(f1s)[::-1][:100])} | Avg threshold -> {np.mean(sorted_thresholds[:100])}")
    import sys; sys.exit()


# Predictions should be the probabilities (i.e. between 0.0 and 1.0)
def precision_recall(predictions, labels, threshold):
    binary_predictions = (predictions > threshold).float()
    # Get true positive and true negatives
    # compute predictions+labels
    # value 2 is TP, value 0 is TN
    preds_plus_labels = binary_predictions + labels
    true_positives = (preds_plus_labels == 2).float()
    true_negatives = (preds_plus_labels == 0).float()
    # Get false positive and false negatives
    # compute predictions-labels
    # value 1 is FP, value -1 is FN
    preds_minus_labels = binary_predictions - labels
    false_positives = (preds_minus_labels == 1).float()
    false_negatives = (preds_minus_labels == -1).float()
    assert torch.all((true_positives + true_negatives + false_positives + false_negatives) == 1), f"TP, TN, FP, FN problem"

    # Micro average: all prediction are equal
    n_tp = torch.sum(true_positives)
    # If we got a zero divided by zero (i.e. a nan) we replace it by 0
    micro_precision = torch.nan_to_num(n_tp / (n_tp + torch.sum(false_positives)), nan=0.).cpu().item()
    micro_recall = torch.nan_to_num(n_tp / (n_tp + torch.sum(false_negatives)), nan=0.).cpu().item()
    micro_f1score = hmean([micro_precision, micro_recall])

    # Macro average: all classes are equal
    n_tp_per_class = torch.sum(true_positives, dim=0)
    assert len(n_tp_per_class) == labels.size(1)
    # If we got a zero divided by zero (i.e. a nan) we replace it by 0
    precision_per_class = torch.nan_to_num(n_tp_per_class / (n_tp_per_class + torch.sum(false_positives, dim=0)), nan=0.)
    macro_precision = torch.mean(precision_per_class).cpu().item()
    recall_per_class = torch.nan_to_num(n_tp_per_class / (n_tp_per_class + torch.sum(false_negatives, dim=0)), nan=0.)
    macro_recall = torch.mean(recall_per_class).cpu().item()
    macro_f1score = hmean([macro_precision, macro_recall])

    return micro_precision, micro_recall, micro_f1score, macro_precision, macro_recall, macro_f1score


def reciprocal_rank(predictions, labels):
    # Sort by highest score, and transform to index directly
    ranked_preds = torch.argsort(predictions, dim=1, descending=True, stable=True)
    ranked_labels = torch.gather(labels, dim=1, index=ranked_preds)
    # Use cumsum to find the first occurrence of a relevant item
    cumulative_relevant = torch.cumsum(ranked_labels, dim=1)
    # Mask out all positions except for the first relevant one (where cumsum is 1)
    first_relevant_mask = (cumulative_relevant == 1)
    ranks = torch.arange(1.0, predictions.size(1) + 1.0).to(predictions.device)
    # Broadcast of ranks into first_relevant_mask
    first_relevant_ranks = ranks * first_relevant_mask
    # For each row, get the first non-zero rank (the rank of the first relevant item)
    # Transform 0.0 to 'inf' so we can use the min function
    first_relevant_ranks[first_relevant_ranks == 0.0] = float('inf')
    reciprocal_ranks, _ = torch.min(first_relevant_ranks, dim=1)
    reciprocal_ranks = 1.0 / reciprocal_ranks

    return torch.mean(reciprocal_ranks, dim=0).cpu().item()


def mean_average_precision_at_k(predictions, labels, k):
    # Sort by highest score, and transform to index directly
    ranked_preds = torch.argsort(predictions, dim=1, descending=True, stable=True)
    # Get the top_k ranked labels per each sample
    top_k = ranked_preds[:, :k]
    top_k_labels = torch.gather(labels, dim=1, index=top_k)
    # Represent the precision level, i.e. 1 to k
    # unsqueeze so that it can be broadcasted
    ranks = torch.arange(1.0, k + 1.0).unsqueeze(0).to(predictions.device)
    # Compute the precision@i
    cumulative_relevant = torch.cumsum(top_k_labels, dim=1)
    precisions_at_i = cumulative_relevant / ranks
    # MAP only considers precisions@i where label == 1
    # precision_mask = top_k_labels
    masked_precisions = precisions_at_i * top_k_labels
    # Average of the precisions at relevant positions, avoid division by 0
    average_precisions_per_sample = torch.sum(masked_precisions, dim=1) / torch.clamp(torch.sum(top_k_labels, dim=1), min=1)

    return torch.mean(average_precisions_per_sample, dim=0).cpu().item()


def ndcg_k_baseline(predictions, labels, k):
    # Sort by highest score, and transform to index directly
    ranked_preds = torch.argsort(predictions, dim=1, descending=True, stable=True)
    # Get the top_k ranked labels per each sample
    top_k = ranked_preds[:, :k]
    top_k_labels = torch.gather(labels, dim=1, index=top_k).cpu().numpy()
    weights = 1/np.log2(np.arange(2,2+k))
    weights = weights[None,:]

    dcg = np.sum( top_k_labels*weights,axis=1)
    ones = np.ones_like(top_k_labels)
    normalization = np.sum(ones*weights,axis=1)
    ndcgs = dcg/normalization

    return np.mean(ndcgs)
