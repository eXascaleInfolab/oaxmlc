# OAXMLC: a Two-Taxonomy Dataset for Benchmarking Extreme Multi-Label Classification

This repository contains the code to run the baselines of the benchmarking of the OAXMLC dataset.

To download the OAXMLC dataset and read the detailed documentation, we refer the reader to the Zenodo repository page of the dataset:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15120226.svg)](https://doi.org/10.5281/zenodo.15120226)

The files `ontology.json`, `taxonomy.txt` and `documents.json`, downloaded from Zenodo, need to be located under a common folder (`datasets/oaxmlc_topics` or `datasets/oaxmlc_concepts`), to run baselines on the dataset with the specific taxonomy. This will make it easier to choose the taxonomy from the `dataset_path` option (see the [config section](#config-options) below), and so that the code runs without ambiguity.

The baselines include:
1. AttentionXML
2. HECTOR
3. MATCH
4. XML-CNN
5. FastXML
6. CascadeXML
7. LightXML
8. Parabel


## Example tasks
OAXMLC's first use-case is benchmarking extreme multi-label classification (XMLC) algorithms as stated in this repository. However, OAXMLC could be reused beyond XMLC experiments as it contains many additional fields (e.g., titles, abstracts, authors, …).  For example, the reference field can be leveraged to build a citation graph, and this graph can be used to e.g., predict missing citations, improve the labeling of documents, or identify clusters of papers, which may help with the detection of trends and emergence of new topics in computer science research.


## Project structure
The folder structure should look like the displayed one. Especially, make sure to put the dataset files downloaded from Zenodo inside the `datasets` folder as shown.

├── algorithms
├── AttentionXML
├── configs
   └── base_config.py
├── datahandler
├── datasets
   ├── oaxmlc_concepts
        └── documents.json
        └── ontology.json
        └── taxonomy.txt
   ├── oaxmlc_topics
        └── documents.json
        └── ontology.json
        └── taxonomy.txt
├── environment.yml
├── .vector_cache
├── FastXML
├── Hector
├── LICENSE
├── LightXML
├── misc
├── models
├── OAXMLC_benchmarking.pdf
├── Parabel
└── README.md


## Config options
This section details the various available options from the `configs/base_config.py` example file.
- **dataset_path**: Path to the dataset. Make sure that the files `ontology.json`, `taxonomy.txt` and `documents.json` are located under a common folder pointed by this path (see example structure above)
- **output_path**: Path to the output folder (automatically created if not existing)
- **exp_name**: Name of the experiment, takes the name of the config file by default. We suggest to left this unchanged to avoid name conflicts
- **device**: Device on which to run the experiment, either `cpu`, `cuda`, or `cuda:x` with `x` a specific GPU number
- **method**: Learning algorithm to use, either `attentionxml`, `hector`, `match`, `xmlcnn`, `cascadexml`, `fastxml`, `lightxml` or `parabel`
- **learning_rate**: Learning rate to use for training
- **seq_length**: Length of the input sequence, i.e. the number of tokens in one input sample
- **voc_size**: (Maximum) Size of the vocabulary
- **tokenization_mode**: How are the texts tokenized, either `word`, `bpe` or `unigram`
- **k_list**: List of *@k* (integers) on which to evaluate the metrics *during training*
- **k_list_eval_perf**: List of *@k* (integers) on which to evaluate the metrics for the final evaluation (e.g. on the test set)
- **hector_params**: Various parameters for HECTOR
	- **loss_smoothing**: Value of the loss smoothing

Other, specific algorithms parameters, such as in `fastxml` and `parabel`, can be modified for the given method under the `algorithms` folder.


## How to run
To handle the packages dependencies and requirements, an `environment.yml` file is provided. A [conda environment](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), after having clone this repository, can thus be created with
```
conda env create -f environment.yml
``` 
Afterwards the virtual environment is activated using
```
conda activate oaxmlc
```

Then, a config file needs to be created inside the `configs` folder, and updated with the desired parameters. See `configs/base_config.py` for an example. The experiment is then launched and executed using
```
python configs/{name_of_the_config_file}.py
```


### FastXML
FastXML needs to be compiled before running. To do so, run the following commands:
```
cd FastXML
python setup.py develop
```

### HECTOR
HECTOR requires also the *GloVe* word embedding. We used the `GloVe.840B.300d` version, which can be downloaded in the [official website](https://nlp.stanford.edu/projects/glove/). The downloaded file must placed in a `.vector_cache` directory by default. The location of the pre-trained word embeddings, a parameter named `path_to_glove`, can be modified in the `algorithms/hector.py` file.
