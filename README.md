# OAXMLC: a Two-Taxonomy Dataset for Benchmarking Extreme Multi-Label Classification
This repository contains the code to run the baselines of the benchmarking of the OAXMLC dataset.

To download the OAXMLC dataset and read the detailed documentation, we refer the reader to the [Zenodo repository page of the dataset](https://zenodo.org/records/15120227).


The files `ontology.json`, `taxonomy.txt` and `documents.json`, downloaded from Zenodo, need to be located under a common folder, to run baselines on the dataset with the specific taxonomy.


We suggest to make one folder for the *topics* taxonomy, and one for the *concepts* taxonomy, as it will be easier to choose the taxonomy from the `dataset_path` option (see the [config section](#config-options) below).

The baselines include:
1. AttentionXML
2. HECTOR
3. MATCH
4. XML-CNN
5. TAMLEC
6. CascadeXML
7. FastXML
8. LightXML
9. Parabel


## How to run
To handle the packages dependencies and requirements, an `environment.yml` file is given. A [conda environment](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), after having clone this repository, can thus be created with
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

### HECTOR and TAMLEC
HECTOR and TAMLEC requires also the *GloVe* word embedding. We used the `GloVe.840B.300d` version, which can be downloaded in the [official website](https://nlp.stanford.edu/projects/glove/). The downloaded file must placed in a `.vector_cache` directory by default. The location of the pre-trained word embeddings, a parameter named `path_to_glove`, can be modified in the `algorithms/hector.py` or `algorithms/tamlec.py` files.


## Config options
This section details the various available options from the `base_config.py` example file.
- **dataset_path**: Path to the dataset. Make sure that the files `ontology.json`, `taxonomy.txt` and `documents.json` are located under a common folder pointed by this path
- **output_path**: Path to the output folder (automatically created if not existing)
- **exp_name**: Name of the experiment, takes the name of the config file by default. We suggest to left this unchanged to avoid name conflicts
- **device**: Device on which to run the experiment, either `cpu`, `cuda`, or `cuda:x` with `x` a specific GPU number
- **method**: Learning algorithm to use, either `attentionxml`, `hector`, `match`, `xmlcnn`, `tamlec`, `cascadexml`, `fastxml`, `lightxml` or `parabel`
- **learning_rate**: Learning rate to use for training
- **seq_length**: Length of the input sequence, i.e. the number of tokens in one input sample
- **voc_size**: (Maximum) Size of the vocabulary
- **tokenization_mode**: How are the texts tokenized, either `word`, `bpe` or `unigram`
- **k_list**: List of *@k* (integers) on which to evaluate the metrics *during training*
- **k_list_eval_perf**: List of *@k* (integers) on which to evaluate the metrics for the final evaluation (e.g. on the test set)
- **tamlec_params**: Various parameters for the HECTOR/TAMLEC methods

Other, specific algorithms parameters, such as in `fastxml` and `parabel`, can be modified for the given method under the `algorithms` folder.
