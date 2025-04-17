from torch.utils.data import DataLoader
import random
import json
import sentencepiece as spm
import numpy as np
import torch
import torchtext.vocab as tvoc
from tqdm import tqdm
import re
import nltk
import copy

from datahandler import datasets
from datahandler.tasks_sampler import SubtreeSampler
from datahandler.global_sampler import collate_global_int_labels
from datahandler.taxonomy import Taxonomy
from datahandler.embeddings import EmbeddingHandler
from misc import utils


# Tools for text pre-processing
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")
stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.snowball.SnowballStemmer('english')


def load_data(cfg):
    """Loads and prepares data for various machine learning algorithms based on the provided configuration.

    This function handles data preprocessing, dataset splitting, embedding generation, and dataloader creation. 
    If preprocessed data is available, it loads the data directly; otherwise, it performs the necessary preprocessing 
    and saves the results for future use.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        dict: A dictionary containing dataloaders and embeddings. Depending on the method, the keys in the 
              dictionary may include:
              - 'global_train': Dataloader for all documents of the training set.
              - 'global_validation': Dataloader for all documents of the validation set.
              - 'global_test': Dataloader for all documents of the test set.
              - 'tasks_train': Dataloader that gives documents for each task, for the training set.
              - 'tasks_validation': Dataloader that gives documents for each task, for the validation set.
              - 'tasks_test': Dataloader that gives documents for each task, for the test set.
              - 'embeddings': Embeddings for the vocabulary. Can be pre-trained or randomly generated.
    """

    # Check if data is already pre-processed
    condition1 = cfg['paths']['taxos_hector'].is_file()
    condition2 = cfg['paths']['taxos_tamlec'].is_file()
    condition3 = cfg['paths']['global_datasets'].is_file()

    if not (condition1 and condition2 and condition3):
        cfg['taxonomy'] = Taxonomy()
        cfg['taxonomy'].load_taxonomy(cfg)
        print(f"> Loading and pre-processing of the data...")
        # Document and list of labels per each sample
        documents = []
        labels = []
        file_names = ['documents.json', 'train.json', 'test.json']
        for file_name in file_names:
            file_path = cfg['paths']['dataset'] / file_name
            if file_path.exists():
                with open(file_path) as f:
                    # Each line is a new sample
                    for idx, line in enumerate(f):
                        data = json.loads(line)
                        # Pre-process the line
                        # Simple pre-processing for some datasets, more advanced for others
                        try:
                            raw_line = data['text_processed']
                            simple = True
                        except KeyError:
                            try:
                                raw_line = data['abstract']
                                simple = False
                            except KeyError:
                                raise NotImplementedError("Check the field of the dataset where the text is")
                        # Simple lowers the text and removes special symbols
                        # Otherwise it also remove stopwords, apply lemmatization
                        preproc_line = text_preprocessing(raw_line, simple=simple)
                        documents.append(preproc_line)
                        # Add the labels
                        ids_labels = []
                        # Depends on the dataset
                        if cfg['dataset'] == 'oatopics':
                            for label in data['topics_labels']:
                                ids_labels.append(label)
                        elif cfg['dataset'] == 'oaconcepts':
                            for label in data['concepts_labels']:
                                ids_labels.append(label)
                        else:
                            assert 'label' in data, f"Key 'label' cannot be found in the ontology, make sure of the key name"
                            for label in data['label']:
                                ids_labels.append(label)
                        # For each label, add all its ancestors to the label set
                        ids_labels = complete_labels(ids_labels, cfg['taxonomy'].label_to_parents)
                        labels.append(ids_labels)
                        if idx % 2000 == 0: print('.', flush=True, end='')
        print()
        assert len(documents) > 0, f"No documents loaded, check filenames ({file_names})"
        assert len(documents) == len(labels), f"Incorrect training data, got {len(documents)} text samples and {len(labels)} labels"

        # Tokenize all documents
        texts_tokenized, vocabulary = tokenization(documents, cfg)
        # Split the data
        global_indices, global_relevant_labels, tasks_indices, tasks_relevant_labels = data_split(texts_tokenized, labels, cfg)
        # Get the embeddings
        emb_handler = EmbeddingHandler(cfg)
        embeddings = emb_handler.get_glove_embeddings(vocabulary, special_embs=True)

        # Save data
        # Data for hector and tamlec
        torch.save(cfg['tamlec_params']['src_vocab'], cfg['paths']['src_vocab'])
        torch.save(cfg['tamlec_params']['trg_vocab'], cfg['paths']['trg_vocab'])
        torch.save(cfg['tamlec_params']['abstract_dict'], cfg['paths']['abstract_dict'])
        torch.save(cfg['tamlec_params']['taxos_hector'], cfg['paths']['taxos_hector'])
        torch.save(cfg['tamlec_params']['taxos_tamlec'], cfg['paths']['taxos_tamlec'])
        # Data for all algorithms
        torch.save(cfg['taxonomy'], cfg['paths']['taxonomy'])
        torch.save(embeddings, cfg['paths']['embeddings'])
        torch.save(cfg['task_to_subroot'], cfg['paths']['task_to_subroot'])
        torch.save(cfg['tasks_size'], cfg['paths']['tasks_size'])
        torch.save(cfg['label_to_tasks'], cfg['paths']['label_to_tasks'])        
        # Save local/task indices
        torch.save((tasks_indices, tasks_relevant_labels), cfg['paths']['tasks_datasets'])
        # Save global indices
        torch.save((global_indices, global_relevant_labels), cfg['paths']['global_datasets'])
        # Draw the tasks of the dataset
        utils.draw_tasks(cfg['tamlec_params']['taxos_tamlec'], cfg['paths']['drawn_tasks'], cfg['dataset'])

    # Load already pre-processed datasets
    else:
        print(f"> Found pre-processed data, load it...")
        if cfg['method'] in ['hector', 'tamlec']:
            # Data for hector and tamlec
            with open(cfg['paths']['src_vocab'], 'rb') as f:
                cfg['tamlec_params']['src_vocab'] = torch.load(f)
            with open(cfg['paths']['trg_vocab'], 'rb') as f:
                cfg['tamlec_params']['trg_vocab'] = torch.load(f)
            with open(cfg['paths']['abstract_dict'], 'rb') as f:
                cfg['tamlec_params']['abstract_dict'] = torch.load(f)
            with open(cfg['paths'][f"taxos_{cfg['method']}"], 'rb') as f:
                cfg['tamlec_params'][f"taxos_{cfg['method']}"] = torch.load(f)
        # Data for all algorithms
        with open(cfg['paths']['global_datasets'], 'rb') as f:
            global_indices, global_relevant_labels = torch.load(f)
        with open(cfg['paths']['tasks_datasets'], 'rb') as f:
            tasks_indices, tasks_relevant_labels = torch.load(f)
        with open(cfg['paths']['taxonomy'], 'rb') as f:
            cfg['taxonomy'] = torch.load(f)
        with open(cfg['paths']['embeddings'], 'rb') as f:
            embeddings = torch.load(f)
        with open(cfg['paths']['task_to_subroot'], 'rb') as f:
            cfg['task_to_subroot'] = torch.load(f)
        with open(cfg['paths']['tasks_size'], 'rb') as f:
            cfg['tasks_size'] = torch.load(f)
        with open(cfg['paths']['label_to_tasks'], 'rb') as f:
            cfg['label_to_tasks'] = torch.load(f)

    # Tasks lengths for HECTOR
    if cfg['tamlec_params']['tasks_size']:
        cfg['tamlec_params']['tasks_size'] = cfg['tasks_size']
    else:
        cfg['tamlec_params']['tasks_size'] = None

    # hector, tamlec, fastxml and parabel require integer labels, other algorithms one-hot labels
    one_hot_labels = cfg['method'] not in ['hector', 'tamlec', 'fastxml', 'parabel']

    # Get the tasks datasets and dataloader
    tasks_train_set = datasets.TasksDataset(tasks_indices['train'], tasks_relevant_labels, one_hot_labels, cfg)
    tasks_val_set = datasets.TasksDataset(tasks_indices['val'], tasks_relevant_labels, one_hot_labels, cfg)
    tasks_test_set = datasets.TasksDataset(tasks_indices['test'], tasks_relevant_labels, one_hot_labels, cfg)

    # Construct samplers
    train_sampler = SubtreeSampler(tasks_train_set, cfg=cfg, batch_size=cfg['batch_size_train'])
    val_sampler = SubtreeSampler(tasks_val_set, cfg=cfg, batch_size=cfg['batch_size_eval'])
    test_sampler = SubtreeSampler(tasks_test_set, cfg=cfg, batch_size=cfg['batch_size_eval'])

    # Special collate function for hector and tamlec
    if cfg['method'] in ['hector', 'tamlec']:
        resampled_train_set = datasets.ResampledTasksDataset(tasks_indices['train'], tasks_relevant_labels, cfg)
        new_sampler = SubtreeSampler(resampled_train_set, cfg=cfg, batch_size=cfg['batch_size_train'])
        tasks_train_loader = DataLoader(resampled_train_set, sampler=new_sampler, collate_fn=new_sampler.collate_hector_tamlec(seed=None))
        # Seed is fixed for validation and test sets to have always the same batches, which is not the case for training
        tasks_val_loader = DataLoader(tasks_val_set, sampler=val_sampler, collate_fn=val_sampler.collate_hector_tamlec(seed=16))
        tasks_test_loader = DataLoader(tasks_test_set, sampler=test_sampler, collate_fn=test_sampler.collate_hector_tamlec(seed=16))
    # Special collate function for fastxml and parabel
    elif cfg['method'] in ['fastxml', 'parabel']:
        tasks_train_loader = DataLoader(tasks_train_set, sampler=train_sampler, collate_fn=train_sampler.collate_no_batch)
        tasks_val_loader = DataLoader(tasks_val_set, sampler=val_sampler, collate_fn=val_sampler.collate_no_batch)
        tasks_test_loader = DataLoader(tasks_test_set, sampler=test_sampler, collate_fn=test_sampler.collate_no_batch)
    # Standard sampler
    else:
        tasks_train_loader = DataLoader(tasks_train_set, sampler=train_sampler, collate_fn=train_sampler.collate_standard(seed=None))
        tasks_val_loader = DataLoader(tasks_val_set, sampler=val_sampler, collate_fn=val_sampler.collate_standard(seed=16))
        tasks_test_loader = DataLoader(tasks_test_set, sampler=test_sampler, collate_fn=test_sampler.collate_standard(seed=16))

    # Get the global datasets and dataloaders
    global_train_set = datasets.GlobalDataset(global_indices['train'], global_relevant_labels, one_hot_labels, cfg, train_dataset=True)
    global_val_set = datasets.GlobalDataset(global_indices['val'], global_relevant_labels, one_hot_labels, cfg, train_dataset=False)
    global_test_set = datasets.GlobalDataset(global_indices['test'], global_relevant_labels, one_hot_labels, cfg, train_dataset=False)
    if not one_hot_labels:
        # Do not shuffle in validation and test sets so we have always the same batches, which is not the case for training
        global_train_loader = DataLoader(global_train_set, batch_size=cfg['batch_size_train'], shuffle=True, drop_last=False, collate_fn=collate_global_int_labels)
        global_val_loader = DataLoader(global_val_set, batch_size=cfg['batch_size_eval'], shuffle=False, drop_last=False, collate_fn=collate_global_int_labels)
        global_test_loader = DataLoader(global_test_set, batch_size=cfg['batch_size_eval'], shuffle=False, drop_last=False, collate_fn=collate_global_int_labels)
    else:
        global_train_loader = DataLoader(global_train_set, batch_size=cfg['batch_size_train'], shuffle=True, drop_last=False)
        global_val_loader = DataLoader(global_val_set, batch_size=cfg['batch_size_eval'], shuffle=False, drop_last=False)
        global_test_loader = DataLoader(global_test_set, batch_size=cfg['batch_size_eval'], shuffle=False, drop_last=False)

    return {
        'global_train': global_train_loader,
        'global_validation': global_val_loader,
        'global_test': global_test_loader,
        'tasks_train': tasks_train_loader,
        'tasks_validation': tasks_val_loader,
        'tasks_test': tasks_test_loader,
        'embeddings': embeddings,
    }


def text_preprocessing(raw_line, simple=True):
    # Lower characters, remove trailing spaces, remove end of line
    preproc_line = raw_line.lower().strip().replace('\n', '')
    # Remove all special characters, keep only letters, numbers and spaces
    preproc_line = re.sub(r"[^\w\s]", '', preproc_line)
    if simple:
        return preproc_line

    else:
        all_words = nltk.tokenize.word_tokenize(preproc_line)
        new_preproc_line = []
        for word in all_words:
            # Remove stopwords
            if word.casefold() in stop_words: continue
            # Stemming
            # word = stemmer.stem(word)
            # Lemmatizing
            word = lemmatizer.lemmatize(word)
            new_preproc_line.append(word.strip())

        new_preproc_line = ' '.join(new_preproc_line).strip()
        return new_preproc_line


# Tokenization with the sentencepiece library
def tokenization(texts, cfg):
    """Tokenization Function

    This function tokenizes a list of texts based on the configuration dictionary. It uses the SentencePiece library for training a tokenizer and processing the texts. The function also handles padding, creates a lookup dictionary for vocabulary, and cleans up intermediate files.

    Args:
        texts (list of str): A list of input text strings to be tokenized.
        cfg (dict): Configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - torch.tensor: A tensor of tokenized and padded sequences with shape `(len(texts), seq_length)`.
            - list of str: The vocabulary created during tokenization.
    """

    # Train the tokenizer
    # -1 for bos and eos tokens since we do not use them in classification
    spm.SentencePieceTrainer.train(sentence_iterator=iter(texts), model_prefix=str(cfg['paths']['tokenizer']).split('.')[0], vocab_size=cfg['voc_size'], character_coverage=1.0, model_type=cfg['tokenization_mode'], unk_id=1, pad_id=0, bos_id=-1, eos_id=-1)

    # Tokenize and get token ids for all data
    print(f"> Tokenizing data...")
    sp = spm.SentencePieceProcessor(model_file=str(cfg['paths']['tokenizer']), out_type=int, num_threads=16)
    texts_encoded = sp.encode(texts)
    texts_tokenized = []
    for sample in texts_encoded:
        # Make the samples of the desired size, with cutting and padding
        sample = sample[:cfg['seq_length']]
        while len(sample) < cfg['seq_length']:
            sample.append(0)
        texts_tokenized.append(sample)

    # Create lookup word to id from the created vocabulary
    word_to_idx = {}
    with open(cfg['paths']['vocabulary']) as f:
        for idx, line in enumerate(f):
            word, _ = line.replace('\n', '').split("\t")
            # SentencePiece adds a meta symbol at the beginning of the words, that is used for decoding purposes
            # remove it so words can be found in pre-trained word embedding during lookup
            if word[0] == '▁' and len(word) > 1 and cfg['tokenization_mode'] == 'word':
                word = word[1:]
            word_to_idx[word] = int(idx)

    # Will be used to get the embeddings
    vocabulary = list(word_to_idx.keys())

    # Source vocab for hector and tamlec
    cfg['tamlec_params']['src_vocab'] = tvoc.vocab(word_to_idx, min_freq=0)

    return texts_tokenized, vocabulary


# In the taxonomy, make sure each sample has all available ancestor nodes of labels
def complete_labels(label_list, label_to_parents):
    """Complete Labels Function

    Expands a list of labels by including all their ancestor labels until reaching the root.

    Args:
        label_list (list): A list of labels (strings) to be expanded.
        label_to_parents (dict): A dictionary mapping each label (string) to a list of its parent labels (strings).

    Returns:
        list: A list of all labels from `label_list` along with their ancestor labels.
    """
    complete_label = set()
    for lab in label_list:
        complete_label.add(lab)
        parents = label_to_parents[lab]
        while 'root' not in parents:
            new_parents = set()
            for parent in parents:
                complete_label.add(parent)
                new_parents = new_parents.union(label_to_parents[parent])
            parents = list(new_parents)

    return list(complete_label)


def data_split(documents, labels_data, cfg):
    """Data Splitting and Pruning Function

    This function processes a dataset by pruning unfrequent labels, splitting the data into train, validation, and test sets, and constructing data structures for hierarchical tasks. It handles taxonomy-based label hierarchies and ensures consistency between labels and data splits.

    Args:
        documents (torch.tensor): A tensor containing the tokenized dataset documents.
        labels_data (list of list of str): A list where each element is a list of labels associated with a document.
        cfg (dict): Configuration dictionary

    Returns:
        tuple: A tuple containing:
            - global_sets (dict of torch.tensor): Whole data split into `train`, `val`, and `test`.
            - global_labels (dict of list of lists): Labels corresponding to global_sets, split into `train`, `val`, and `test`.
            - global_relevant_labels (torch.tensor): Relevant labels for the global_sets.
            - all_tasks_sets (dict of list of torch.tensor): Dataset split into `train`, `val`, and `test`, and then by tasks.
            - all_tasks_labels (dict of list of lists of lists): Labels corresponding to all_tasks_sets, split into `train`, `val`, and `test`, and then, by tasks.
            - all_tasks_relevant_labels (list of torch.tensor): Relevant labels for each task.
    """

    print(f"> Get some stats on dataset")
    print(f">> Dataset has {len(documents)} documents")
    taxonomy = cfg['taxonomy']
    # key: label, values: list of items index having this label
    items_per_label = {}
    # key: index, values: list of labels 
    labels_per_idx = {}
    # Get all labels from all items
    for idx, labels in enumerate(labels_data):
        # Each single label for an item
        for label in labels:
            if label in items_per_label:
                items_per_label[label].append(idx)
            else:
                items_per_label[label] = [idx]
            labels_per_idx[idx] = labels
    print(f">> Documents per label: {np.mean([len(items) for items in items_per_label.values()])}")
    print(f">> Labels per document: {np.mean([len(labels) for labels in labels_per_idx.values()])}")

    # Recursively remove leaves in the taxonomy that do not appear in the dataset (otherwise we could have a KeyError afterwards)
    print(f"> Removing leaves not appearing in dataset...")
    leaves_to_remove = [leaf for leaf in taxonomy.leaves if leaf not in [node for node in items_per_label.keys() if taxonomy.is_leaf(node)]]
    while(len(leaves_to_remove) != 0):
        taxonomy.remove_leaves(leaves_to_remove)
        leaves_to_remove = [leaf for leaf in taxonomy.leaves if leaf not in [node for node in items_per_label.keys() if taxonomy.is_leaf(node)]]

    # Prune all leaves until they all have at least min_freq samples
    min_freq = 50
    n_before = len(taxonomy.all_children('root'))
    while(min([len(items_per_label[leaf]) for leaf in taxonomy.leaves])) < min_freq:
        for leaf in taxonomy.leaves:
            if len(items_per_label[leaf]) < min_freq:
                taxonomy.remove_leaves(leaf)
                del items_per_label[leaf]

    level = 1
    # Remove leaves on level 1
    taxonomy.remove_leaves([node for node in taxonomy.level_to_labels[level] if taxonomy.is_leaf(node)])
    all_labels = taxonomy.all_children('root')
    print(f"> Pruned {n_before - len(all_labels)} leaf nodes having less than {min_freq} samples and the leaves on first level")
    print(f">> Updated taxonomy has {taxonomy.n_nodes} nodes and {len(taxonomy.leaves)} leaves and a height of {taxonomy.height}")

    # Take sub-trees only from level 1, if they are not leaves
    # Each sub-tree will be considered as a task
    subtrees_to_keep = [node for node in taxonomy.level_to_labels[level] if node not in taxonomy.leaves]
    # Sorted to make sure sub-trees have always the same order
    subtrees_to_keep = sorted(subtrees_to_keep)
    print(f"> Possible sub-trees {len(subtrees_to_keep)} with root on level {level}")

    # Keep samples that have at least one label
    print(f"> Remove documents having no more label in common with taxonomy")
    indices_to_keep = []
    for idx, labels in labels_per_idx.items():
        doc_labels = [lab for lab in labels if lab in all_labels]
        # If no labels in common with taxonomy
        if len(doc_labels) == 0:
            continue
        # If single label is a root of a sub-tree
        elif (len(doc_labels) == 1) and (doc_labels[-1] in subtrees_to_keep):
            continue
        else:
            indices_to_keep.append(idx)
    indices_to_keep = torch.tensor(indices_to_keep, dtype=torch.int64)

    # Update data: remove documents that have no more labels
    # Update labels: remove labels that are not in taxonomy anymore
    n_docs_before = len(documents)
    new_documents = []
    new_labels = []
    for idx in indices_to_keep:
        # Sorted to always keep the same order
        new_documents.append(documents[idx])
        new_labels.append(sorted(list(set(all_labels).intersection(labels_data[idx]))))
    print(f">> Documents kept: {len(new_documents)} (out of {n_docs_before})")
    assert len(new_documents) == len(new_labels), f"Error in documents removing"

    # Update the data-structures after pruning and cleaning
    # key: label, values: set of items index having this label
    print(f"> Updating storage and stats...")
    items_per_label = {}
    # key: index, values: set of labels
    labels_per_idx = {} 
    # Get all labels from all items
    for idx, labels in enumerate(new_labels):
        # Each single label for an item
        for label in labels:
            if label in items_per_label:
                items_per_label[label].append(idx)
            else:
                items_per_label[label] = [idx]
            labels_per_idx[idx] = labels
    print(f">> Now {np.mean([len(items) for items in items_per_label.values()])} documents per label")
    print(f">> Now {np.mean([len(labels) for labels in labels_per_idx.values()])} labels per document")

    # Construct specific data for hector and tamlec
    # Target vocab, deepcopy to have another reference in memory
    voc_plus_pad = copy.deepcopy(taxonomy.label_to_idx)
    voc_plus_pad['<pad>'] = len(taxonomy.label_to_idx)
    cfg['tamlec_params']['trg_vocab'] = tvoc.vocab(voc_plus_pad, min_freq=0)
    # Taxonomy for hector is full taxonomy
    taxo_id_root = taxonomy.label_to_idx['root']
    children = {taxo_id_root: [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children['root']]}
    for curr_node in taxonomy.all_children('root'):
        if not taxonomy.is_leaf(curr_node):
            children[taxonomy.label_to_idx[curr_node]] = [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[curr_node]]
    cfg['tamlec_params']['taxos_hector'] = [(taxo_id_root, children)]
    # Taxonomies for tamlec are taxonomies of selected sub-trees
    cfg['tamlec_params']['taxos_tamlec'] = []
    for subtree_root in subtrees_to_keep:
        taxo_id_root = taxonomy.label_to_idx[subtree_root]
        children = {taxo_id_root: [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[subtree_root]]}
        for curr_node in taxonomy.all_children(subtree_root):
            if not taxonomy.is_leaf(curr_node):
                children[taxonomy.label_to_idx[curr_node]] = [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[curr_node]]
        cfg['tamlec_params']['taxos_tamlec'].append((taxo_id_root, children))
    # Abstract dict
    cfg['tamlec_params']['abstract_dict'] = {}
    for label, abstract in taxonomy.label_to_abstract.items():
        cfg['tamlec_params']['abstract_dict'][taxonomy.label_to_idx[label]] = abstract

    print(f">> Constructing the labels...")
    final_labels = []
    for labels in new_labels:
        # Add root of taxonomy
        class_list = [0] + [taxonomy.label_to_idx[label] for label in labels]
        final_labels.append(class_list)

    # Various data
    cfg['tasks_size'] = []
    cfg['task_to_subroot'] = {}
    # Root belongs to no task
    cfg['label_to_tasks'] = {taxonomy.label_to_idx['root']: set()}
    # Dict of document indices per split
    global_indices = {'train': set(), 'val': set(), 'test': set()}
    # Dict of lists (by split) of lists (by tasks) of indices (integer)
    all_tasks_indices = {'train': [], 'val': [], 'test': []}
    # List of lists of relevant labels for each task
    all_tasks_relevant_labels = []
    # Set of all possible labels (root and sub-roots excluded)
    global_relevant_labels = set()

    print(f">> Train-val-test split...")
    for subtree_idx, subtree_root in enumerate(tqdm(subtrees_to_keep, leave=False)):
        # Dict of indices for this task
        task_indices = {'train': set(), 'val': set(), 'test': set()}
        all_subtree_indices = set()
        relevant_labels_task = []
        # Set idx of the node that is the root of this sub-tree
        cfg['task_to_subroot'][subtree_idx] = taxonomy.label_to_idx[subtree_root]
        try:
            cfg['label_to_tasks'][taxonomy.label_to_idx[subtree_root]].add(subtree_idx)
        except KeyError:
            cfg['label_to_tasks'][taxonomy.label_to_idx[subtree_root]] = set([subtree_idx])
            
        # Make the split for all other nodes in the sub-tree
        for subnode in taxonomy.all_children(subtree_root):
            # Subnode to the task
            try:
                cfg['label_to_tasks'][taxonomy.label_to_idx[subnode]].add(subtree_idx)
            except KeyError:
                cfg['label_to_tasks'][taxonomy.label_to_idx[subnode]] = set([subtree_idx])
            # Add node to the relevant indices of this task and to the global
            relevant_labels_task.append(taxonomy.label_to_idx[subnode])
            global_relevant_labels.add(taxonomy.label_to_idx[subnode])
            # Get document indices of this node
            indices_in_subnode = set(items_per_label[subnode])
            # Complete document indices of this task
            all_subtree_indices = all_subtree_indices.union(indices_in_subnode)
            # Update task indices if already in global indices
            task_indices['test'] = task_indices['test'].union(global_indices['test'].intersection(indices_in_subnode))
            task_indices['val'] = task_indices['val'].union(global_indices['val'].intersection(indices_in_subnode))
            task_indices['train'] = task_indices['train'].union(global_indices['train'].intersection(indices_in_subnode))
            # Sorted to make sure it has always the same order
            documents_not_seen = sorted(list(indices_in_subnode.difference(task_indices['test']).difference(task_indices['val']).difference(task_indices['train'])))
            if len(documents_not_seen) == 0:
                continue

            random.seed(42)
            random.shuffle(documents_not_seen)
            # Train 70% - validation 15% - test 15%
            # Make sure the sets are mutually exclusive
            train_idx = set(documents_not_seen[:int(0.7*len(documents_not_seen))])
            val_idx = set(documents_not_seen[int(0.7*len(documents_not_seen)):int(0.85*len(documents_not_seen))])
            test_idx = set(documents_not_seen[int(0.85*len(documents_not_seen)):])
            # Update task and global indices
            task_indices['train'] = task_indices['train'].union(train_idx)
            global_indices['train'] = global_indices['train'].union(train_idx)
            task_indices['val'] = task_indices['val'].union(val_idx)
            global_indices['val'] = global_indices['val'].union(val_idx)
            task_indices['test'] = task_indices['test'].union(test_idx)
            global_indices['test'] = global_indices['test'].union(test_idx)

        # Make sure the indices are mutually exclusive within the task
        assert (not task_indices['train'].intersection(task_indices['val'])) and (not task_indices['train'].intersection(task_indices['test'])) and (not task_indices['val'].intersection(task_indices['test'])), f"Tasks sets are not mutually exclusive"
        assert (not global_indices['train'].intersection(global_indices['val'])) and (not global_indices['train'].intersection(global_indices['test'])) and (not global_indices['val'].intersection(global_indices['test'])), f"Global sets are not mutually exclusive"
        # Make sure all samples were taken
        assert len(task_indices['train']) + len(task_indices['val']) + len(task_indices['test']) == len(all_subtree_indices)

        # Get indices of this task
        for split, set_indices in task_indices.items():
            all_tasks_indices[split].append(sorted(list(set_indices)))
        relevant_labels_task = torch.tensor(relevant_labels_task, dtype=torch.int64)
        all_tasks_relevant_labels.append(relevant_labels_task)

        # Number of elements in each task
        cfg['tasks_size'].append(len(all_subtree_indices))

    # Make sure the sets are mutually exclusive
    assert (not global_indices['train'].intersection(global_indices['val'])) and (not global_indices['train'].intersection(global_indices['test'])) and (not global_indices['val'].intersection(global_indices['test'])), f"Sets are not mutually exclusive"
    print(f">> Unique documents: {len(global_indices['train']) + len(global_indices['val']) + len(global_indices['test'])}")
    print(f">> Training set: {len(global_indices['train'])}, validation set: {len(global_indices['val'])}, test set: {len(global_indices['test'])}")

    print(f"> Constructing global sets...")
    global_relevant_labels = torch.tensor(list(global_relevant_labels), dtype=torch.int64)
    # Save global indices, sorted so it is deterministic
    for split in global_indices.keys():
        global_indices[split] = sorted(list(global_indices[split]))

    # Save the tokenized documents and their labels into separate files
    # This will be loaded at runtime
    print(f"> Saving documents...")
    for indices in global_indices.values():
        for doc_idx in tqdm(indices, leave=False):
            tokenized_doc = new_documents[doc_idx]
            labels = final_labels[doc_idx]
            file_path = cfg['paths']['data'] / f"{doc_idx}.json"
            assert not file_path.exists()
            with open(file_path, 'w') as f:
                json.dump((tokenized_doc, labels), f)
    assert len(list(cfg['paths']['data'].iterdir())) == len(global_indices['train']) + len(global_indices['val']) + len(global_indices['test']), "Why mismatch?"

    return global_indices, global_relevant_labels, all_tasks_indices, all_tasks_relevant_labels
