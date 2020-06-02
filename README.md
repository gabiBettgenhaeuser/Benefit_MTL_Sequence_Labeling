## Structure of Supplementary Material
This repository has the following structure:

* code
1. LSTM_0FCs.py; LSTM_1FCs.py; LSTM_2FCs.py; LSTM_3FCs.py: implementation of 4 different multitask models in Pytorch for any number of tasks. 
2. main.py: code for loading the real and synthetic tasks and for the experimental pipeline of training and testing MTL models.
3. utilities.py: utility functions for loading and preprocessing CoNLL dataset, word embeddings, model evaluation, task simulator, computation of the adjusted mutual information (AMI) and visualizations. 
4. experimentalsettings.py: utility code to store experimental configurations.
5. Symbolic Regression_Balanced Datasets.ipynb; Symbolic Regression_Unbalanced Datasets.ipynb: notebooks with the experimental pipeline for applying symbolic regression. 

* config
1. config.json: configurations to control the parameters of the task simulator, hyperparameters of the models, paths to embeddings, paths to store outputs and tagging schemes. 

* data
1. conll2003: Named Entity Recognition dataset base for Task Simulator and for fitting/evaluating MTL models. Should be obtained from the authors.
2. MTL Balanced Datasets.csv; MTL Unbalanced Datasets.csv: generated data to fit the symbolic regressors.

## Installation
```
# Execute in this directory
conda create --name benefit-mtl python=3.7.6
source activate benefit-mtl
pip install torch==1.4.0 gplearn==0.4.1 matplotlib==3.1.1 tqdm==4.41.1 numpy==1.17.4 gensim==3.8.0 scikit-learn==0.22.1 pandas==1.0.1 seaborn==0.10.0 ipython==7.11.1
# For the training and evaluation of the MTL models, the dataset CoNLL needs to be obtained from the original authors.
``` 

## Running

* Config Files:
The values for each key of the JSON files should have the following values: \
{\
	"NAME": if the name of the JSON file is XXX.json, then "XXX"  should be the value of "NAME" \
	"MODEL_NAME": one of the options in {"LSTM_0FCs", "LSTM_1FCs", "LSTM_2FCs", "LSTM_3FCs"} \
	"TEST_TYPE": one of the options in {"TASK_NUMBER", "TASK_CORRELATION", "NUMBER_TOKENS"} \
	"EMBEDDING": one of the options in {"FASTTEXT", "GLOVE"},
	"LOG_PATH": "/path/to/log" \
	"PLOT_SCORES_PATH": "/path/to/plot.png" \
	"PLOT_LABELCOOCURRENCE_PATH": "/path/to/ploy2" \
	"PATH_TRAIN": "/path/to/train-data" \
	"PATH_DEV": "/path/to/dev-data" \
	"PATH_TEST": "/path/to/dev-data" \
	"PATH_FASTTEXT": "/path/to/emb" \
	"PATH_GLOVE": "/path/to/emb" \
	"CONTEXT_SIZE": int \
	"EMBEDDING_DIM": int \
	"HIDDEN_SIZE": int \
	"BATCH_SIZE": int \
	"EPOCHS": int \
	"OUTPUT_SIZE": int \
	"TASK_NUMBER": int \
	"NUMBER_TOKENS": int \
	"ALPHA": int \
	"ITERATIONS": int \
	"TAGGING_SCHEME_TRAIN": one of the options in {"IO", "BIO", "BINARY"} \
	"TAGGING_SCHEME_DEV": one of the options in {"IO", "BIO", "BINARY"} \
	"TAGGING_SCHEME_TEST": one of the options in {"IO", "BIO", "BINARY"}.	
}


* Running experiments:
#For training and evaluating the MTL models
python main.py

#For training and evaluating the symbolic regressors
jupyter notebook
#search for the files and run the notebook

```
The resulting plots and log files are stored in the directories as specified in the JSON file. 
