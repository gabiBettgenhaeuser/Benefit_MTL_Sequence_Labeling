import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging
import utilities
import LSTM_3FCs
import LSTM_2FCs
import LSTM_1FCs
import LSTM_0FCs
import matplotlib.pyplot as plt
import matplotlib
import experimentalsettings as ExperimentalSettings

# Loading JSON with config file
SETTINGS = ExperimentalSettings.ExperimentalSettings.load_json("MTL_ALPHA_LSTM_1FCs_DEPLOY_CLUSTER")

# Setting log information
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s: %(message)s')
file_handler = logging.FileHandler(SETTINGS["LOG_PATH"])
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Setting Device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

logger.info("\n\n\n")
logger.info('\nModel: {}\nIterations: {}\nNum_tasks: {}\nAmount data: {}\nCorrelation: {}\nBatch size: {}\nNum_epochs: {}'.format(SETTINGS["MODEL_NAME"], SETTINGS["ITERATIONS"], SETTINGS["TASK_NUMBER"], SETTINGS["NUMBER_TOKENS"], SETTINGS["ALPHA"], SETTINGS["BATCH_SIZE"], SETTINGS["EPOCHS"]))

# Loading word embeddings
word_emb = utilities.WordEmbedding()
if SETTINGS["EMBEDDING"] == "FASTTEXT":
    emb = word_emb.load_fasttext(SETTINGS["PATH_FASTTEXT"])
elif SETTINGS["EMBEDDING"] == "GLOVE":
    emb = word_emb.load_glove(SETTINGS["PATH_GLOVE"])
print('end_embedding')


def prepare_data(path_to_data, label_type, dataset_type):
    # Load corpora
    data_creation = utilities.Corpora()
    words, labels = data_creation.load_conll_dataset(path_to_data)

    # Prepare x in context
    sentence = utilities.Sentence(SETTINGS["CONTEXT_SIZE"])
    instances = sentence.input_in_context(words, labels)

    # embed words in vector representation
    x = word_emb.instances_to_embedding(instances)
    x = x.clone().detach()

    # Preprocess labels
    y = []
    lb = utilities.LabelRepresentation()
    if label_type == 'BINARY':
        lb.use_ner_binary_labels()
        for i in range(len(instances)):
            instances[i].label[0] = lb.NER_binary(instances[i].label[0])
            y.append(lb.label_to_idx(instances[i].label[0]))
        if dataset_type != 'TRAIN':
            temp_labels = []
            for instance in instances:
                temp_labels.append(instance.label[0])
            temp_labels = utilities.LabelRepresentation.convert_noprefix_to_bio_labels(temp_labels)
            for cnt, instance in enumerate(instances):
                instance.label[0] = temp_labels[cnt]
    elif label_type == 'IO':
        lb.use_ner_noprefix_labels()
        for i in range(len(instances)):
            instances[i].label[0] = lb.NER_without_prefix(instances[i].label[0])
            y.append(lb.label_to_idx(instances[i].label[0]))
    elif label_type == 'BIO':
        lb.use_ner_bio_labels()
        for i in range(len(instances)):
            y.append(lb.label_to_idx(instances[i].label[0]))
    y = torch.tensor(y, dtype=torch.long)
    # Synthetic data:
    temp_y = y.view(-1, 1)
    sd = utilities.SyntheticData()
    for n_tasks in range(SETTINGS["TASK_NUMBER"] - 1):
        s_y = sd.generate_data(y, list(lb.idx_to_label_map.keys()), SETTINGS["ALPHA"][n_tasks])
        s_y = s_y.view(-1, 1)
        temp_y = torch.cat((temp_y, s_y), dim=1)
        if label_type == 'BINARY':
            lb.use_synthetic_binary_labels()
            temp_labels = []
            for idx, label in enumerate(temp_y[:, n_tasks + 1]):
                temp_labels.append(label.item())
                temp_labels[idx] = lb.idx_to_label(temp_labels[idx])
        elif label_type == 'IO':
            lb.use_synthetic_noprefix_labels()
            temp_labels = []
            for idx, label in enumerate(temp_y[:, n_tasks + 1]):
                temp_labels.append(label.item())
                temp_labels[idx] = lb.idx_to_label(temp_labels[idx])
        elif label_type == 'BIO':
            lb.use_synthetic_bio_labels()
            temp_labels = []
            for idx, label in enumerate(temp_y[:, n_tasks + 1]):
                temp_labels.append(label.item())
                temp_labels[idx] = lb.idx_to_label(temp_labels[idx])
        for idx, instance in enumerate(instances):
            instance.label.insert(1, temp_labels[idx])  # note always adding synthetic right after NER
    return instances, x, temp_y

# Load data
train, train_x, train_y = prepare_data(SETTINGS['PATH_TRAIN'], SETTINGS["TAGGING_SCHEME_TRAIN"], 'TRAIN')
dev, dev_x, dev_y = prepare_data(SETTINGS['PATH_DEV'], SETTINGS["TAGGING_SCHEME_DEV"], 'DEV')
test, test_x, test_y = prepare_data(SETTINGS['PATH_TEST'], SETTINGS["TAGGING_SCHEME_TEST"], 'TEST')
print('end loading data')
del emb

# Labels Mutual Information and Coocurrence Matrix
if SETTINGS["TASK_NUMBER"] > 1:
    mi = utilities.MutualInformation(train_y, SETTINGS["TASK_NUMBER"])
    print(mi)
    utilities.LabelCoocurrencePlot(train_y, SETTINGS["TAGGING_SCHEME_TRAIN"], SETTINGS["TASK_NUMBER"], SETTINGS["PLOT_LABELCOOCURRENCE_PATH"])

# Number of steps to unroll
seq_size = 2 * SETTINGS["CONTEXT_SIZE"] + 1

# Loading test type
VAR_TEST = []
if SETTINGS["TEST_TYPE"] == "NUMBER_TOKENS":
    dim_y = len(SETTINGS["NUMBER_TOKENS"])
    VAR_TEST = SETTINGS["NUMBER_TOKENS"]
    xlabel = 'number_tokens'
    n_tasks=SETTINGS["TASK_NUMBER"]
elif SETTINGS["TEST_TYPE"] == "TASK_CORRELATION":
    dim_y = len(SETTINGS["ALPHA"])
    VAR_TEST = SETTINGS["ALPHA"]
    xlabel = 'alpha'
    n_tasks = 2
elif SETTINGS["TEST_TYPE"] == "TASK_NUMBER":
    dim_y = len(list(range(SETTINGS["TASK_NUMBER"])))
    VAR_TEST = list(range(1, SETTINGS["TASK_NUMBER"] + 1))
    xlabel = 'number_tasks'

precision_dev = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
recall_dev = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
fscore_dev = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
precision_test = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
recall_test = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
fscore_test = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)

precision_dev_sd = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
recall_dev_sd = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
fscore_dev_sd = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
precision_test_sd = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
recall_test_sd = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)
fscore_test_sd = torch.zeros(SETTINGS["TASK_NUMBER"], dim_y)

temp_precision_dev = torch.zeros(SETTINGS["TASK_NUMBER"], SETTINGS["ITERATIONS"])
temp_recall_dev = torch.zeros(SETTINGS["TASK_NUMBER"], SETTINGS["ITERATIONS"])
temp_fscore_dev = torch.zeros(SETTINGS["TASK_NUMBER"], SETTINGS["ITERATIONS"])
temp_precision_test = torch.zeros(SETTINGS["TASK_NUMBER"], SETTINGS["ITERATIONS"])
temp_recall_test = torch.zeros(SETTINGS["TASK_NUMBER"], SETTINGS["ITERATIONS"])
temp_fscore_test = torch.zeros(SETTINGS["TASK_NUMBER"], SETTINGS["ITERATIONS"])

# Keeping dev and test set fixed
dev_dataset = TensorDataset(dev_x, dev_y)
test_dataset = TensorDataset(test_x, test_y)

dev_loader = torch.utils.data.DataLoader(dev_dataset, shuffle=False, batch_size=SETTINGS["BATCH_SIZE"], drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=SETTINGS["BATCH_SIZE"],  drop_last=True)

for cnt, value in tqdm(enumerate(VAR_TEST)):
    if SETTINGS["TEST_TYPE"] == "TASK_NUMBER":
        n_tasks = value
    for j in range(SETTINGS["ITERATIONS"]):
        print('{} = {}.'.format(SETTINGS["TEST_TYPE"], value))
        logger.info('{} = {}.'.format(SETTINGS["TEST_TYPE"], value))
		
        # Create model
        if SETTINGS["MODEL_NAME"] == "LSTM_3FCs":
            myModel = LSTM_3FCs.Net(input_size=SETTINGS["EMBEDDING_DIM"], hidden_size=SETTINGS["HIDDEN_SIZE"], output_size=SETTINGS["OUTPUT_SIZE"],
                                    num_tasks= n_tasks, batch_size=SETTINGS["BATCH_SIZE"])
        elif SETTINGS["MODEL_NAME"] == "LSTM_2FCs":
            myModel = LSTM_2FCs.Net(input_size=SETTINGS["EMBEDDING_DIM"], hidden_size=SETTINGS["HIDDEN_SIZE"], output_size=SETTINGS["OUTPUT_SIZE"],
                                    num_tasks= n_tasks, batch_size=SETTINGS["BATCH_SIZE"])
        elif SETTINGS["MODEL_NAME"] == "LSTM_1FCs":
            myModel = LSTM_1FCs.Net(input_size=SETTINGS["EMBEDDING_DIM"], hidden_size=SETTINGS["HIDDEN_SIZE"], output_size=SETTINGS["OUTPUT_SIZE"],
                                    num_tasks= n_tasks, batch_size=SETTINGS["BATCH_SIZE"])
        elif SETTINGS["MODEL_NAME"] == "LSTM_0FCs":
            myModel = LSTM_0FCs.Net(input_size=SETTINGS["EMBEDDING_DIM"], hidden_size=SETTINGS["HIDDEN_SIZE"], output_size=SETTINGS["OUTPUT_SIZE"],
                                    num_tasks= n_tasks, batch_size=SETTINGS["BATCH_SIZE"])

        optimizer = torch.optim.Adam(myModel.parameters())
        criterion = nn.CrossEntropyLoss()
        myModel.to(DEVICE)

        start = 0
        if SETTINGS["TEST_TYPE"] == "NUMBER_TOKENS": #CHANGED
            n_tokens = value
        elif SETTINGS["TEST_TYPE"] == "TASK_CORRELATION" or SETTINGS["TEST_TYPE"] == "TASK_NUMBER":
            n_tokens = SETTINGS["NUMBER_TOKENS"]

        end = n_tokens
        train_datasets = []
        for k in range(n_tasks): #CHANGED
            if SETTINGS["TEST_TYPE"] == "NUMBER_TOKENS": #CHANGED
                y_training = train_y[start:end]
            elif SETTINGS["TEST_TYPE"] == "TASK_CORRELATION":
                y_training = train_y[start:end, [0, cnt+1]]
            elif SETTINGS["TEST_TYPE"] == "TASK_NUMBER":
                y_training = train_y[start:end, 0:n_tasks]
            train_datasets.append(utilities.SequenceLabelingDataset(train_x[start:end], y_training, n_tokens, k))
            start += n_tokens #CHANGED
            end += n_tokens

        train_loaders = []
        for k in range(n_tasks):
            train_loaders.append(
                torch.utils.data.DataLoader(train_datasets[k], shuffle=True, batch_size=SETTINGS["BATCH_SIZE"], drop_last=True))
				
        myModel.train()
        for epoch in tqdm(range(SETTINGS["EPOCHS"])):
            for idx_loader in range(n_tasks):
                for i, (inputs, labels) in enumerate(train_loaders[idx_loader]):
                    loss = 0
                    # 1. Clear gradients w.r.t. parameters
                    optimizer.zero_grad()

                    # 2. Load sentences as a torch tensor with gradient accumulation
                    # inputs = inputs.view(-1, seq_size, SETTINGS["EMBEDDING_DIM"]).requires_grad_()
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                    # 3. Forward pass to get output/logits
                    outputs = myModel(inputs)  # Change need to update model to return a tensor of shape (batch size, #tasks)

                    # 4 Calculate Loss: softmax --> cross entropy loss
                    loss = loss + criterion(outputs[idx_loader], labels)

                    # 5.  Getting gradients w.r.t. parameters
                    loss.backward()

                    # 6. Updating parameters
                    optimizer.step()

        myModel.eval()
        dev_pred_labels = [[] for i in range(n_tasks)] #CHANGED
        test_pred_labels = [[] for i in range(n_tasks)] #CHANGED

        dev_pred = utilities.predict(myModel, dev_loader, seq_size, SETTINGS["EMBEDDING_DIM"], DEVICE, n_tasks) #CHANGED

        lb = []
        for k in range(n_tasks):
            lb.append(utilities.LabelRepresentation())
            if k == 0:
                if SETTINGS["TAGGING_SCHEME_DEV"] == "BIO" or SETTINGS["TAGGING_SCHEME_DEV"] == "IO":
                    lb[k].use_ner_noprefix_labels()
                if SETTINGS["TAGGING_SCHEME_DEV"] == "BINARY":
                    lb[k].use_ner_binary_labels()
            else:
                if SETTINGS["TAGGING_SCHEME_DEV"] == "BIO" or SETTINGS["TAGGING_SCHEME_DEV"] == "IO":
                    lb[k].use_synthetic_noprefix_labels()
                if SETTINGS["TAGGING_SCHEME_DEV"] == "BINARY":
                    lb[k].use_synthetic_binary_labels()

        for k in range(n_tasks): #CHANGED
            for pred in dev_pred[k]:
                dev_pred_labels[k].append(lb[k].idx_to_label((pred.item())))
            dev_pred_labels[k] = lb[k].convert_noprefix_to_bio_labels(dev_pred_labels[k])
            evaluation = utilities.Evaluation()
            if SETTINGS["TEST_TYPE"] == "NUMBER_TOKENS" or SETTINGS["TEST_TYPE"] == "TASK_NUMBER":
                index_string = k
            elif SETTINGS["TEST_TYPE"] == "TASK_CORRELATION":
                if k == 0: #Changed
                    index_string = 0
                else:
                    index_string = cnt+1
            conll_evaluation_string = evaluation.create_conlleval_string(dev, dev_pred_labels[k], idx_label= index_string) #Changed
            overall_dev, per_label_dev, full_report_dev = evaluation.evaluate_conlleval_string(conll_evaluation_string)
            logger.info("Task {}".format(k))
            logger.info("Dev")
            logger.info(full_report_dev)
            temp_precision_dev[k, j] = overall_dev.prec
            temp_recall_dev[k, j] = overall_dev.rec
            temp_fscore_dev[k, j] = overall_dev.fscore

        test_pred = utilities.predict(myModel, test_loader, seq_size, SETTINGS["EMBEDDING_DIM"], DEVICE, n_tasks) #Changed
		
        for k in range(n_tasks):#Changed
            for pred in test_pred[k]:
                test_pred_labels[k].append(lb[k].idx_to_label((pred.item())))
            test_pred_labels[k] = lb[k].convert_noprefix_to_bio_labels(test_pred_labels[k])
            evaluation = utilities.Evaluation()
            if SETTINGS["TEST_TYPE"] == "NUMBER_TOKENS" or SETTINGS["TEST_TYPE"] == "TASK_NUMBER":
                index_string = k
            elif SETTINGS["TEST_TYPE"] == "TASK_CORRELATION":
                if k == 0: #Changed
                    index_string = 0
                else:
                    index_string = cnt+1
            conll_evaluation_string = evaluation.create_conlleval_string(test, test_pred_labels[k], idx_label=index_string)
            overall_test, per_label_test, full_report_test = evaluation.evaluate_conlleval_string(
                conll_evaluation_string)
            logger.info("Test")
            logger.info(full_report_test)
            temp_precision_test[k, j] = overall_test.prec
            temp_recall_test[k, j] = overall_test.rec
            temp_fscore_test[k, j] = overall_test.fscore
        del myModel
		
    for k in range(n_tasks): #CHANGED
        precision_dev[k, cnt] = temp_precision_dev[k].mean()
        recall_dev[k, cnt] = temp_recall_dev[k].mean()
        fscore_dev[k, cnt] = temp_fscore_dev[k].mean()

        precision_test[k, cnt] = temp_precision_test[k].mean()
        recall_test[k, cnt] = temp_recall_test[k].mean()
        fscore_test[k, cnt] = temp_fscore_test[k].mean()

        precision_dev_sd[k, cnt] = temp_precision_dev[k].std()
        recall_dev_sd[k, cnt] = temp_recall_dev[k].std()
        fscore_dev_sd[k, cnt] = temp_fscore_dev[k].std()

        precision_test_sd[k, cnt] = temp_precision_test[k].std()
        recall_test_sd[k, cnt] = temp_recall_test[k].std()
        fscore_test_sd[k, cnt] = temp_fscore_test[k].std()

for k in range(n_tasks):
    print("\nTask {}".format(k))
    print("\nDevelopment set")
    print('Prec: {}\nRecall: {}\nFscore: {}'.format(precision_dev[k], recall_dev[k], fscore_dev[k]))
    print('Std_Prec: {}\nStd_Recall: {}\nStd_Fscore: {}'.format(precision_dev_sd[k], recall_dev_sd[k], fscore_dev_sd[k]))
    print("\nTesting set")
    print('Prec: {}\nRecall: {}\nFscore: {}'.format(precision_test[k], recall_test[k], fscore_test[k]))
    print('Std_Prec: {}\nStd_Recall: {}\nStd_Fscore: {}'.format(precision_test_sd[k], recall_test_sd[k], fscore_test_sd[k]))
    logger.info("\nTask {}".format(k))
    logger.info("\nDevelopment set")
    logger.info('Prec: {}\nRecall: {}\nFscore: {}'.format(precision_dev[k], recall_dev[k], fscore_dev[k]))
    logger.info('Std_Prec: {}\nStd_Recall: {}\nStd_Fscore: {}'.format(precision_dev_sd[k], recall_dev_sd[k], fscore_dev_sd[k]))
    logger.info("\nTesting set")
    logger.info('Prec: {}\nRecall: {}\nFscore: {}'.format(precision_test[k], recall_test[k], fscore_test[k]))
    logger.info('Std_Prec: {}\nStd_Recall: {}\nStd_Fscore: {}'.format(precision_test_sd[k], recall_test_sd[k], fscore_test_sd[k]))

if SETTINGS["TASK_NUMBER"] > 1:
    logger.info("\nMutual Information: {}".format(mi))

# Result Plots
matplotlib.use('Agg')

colors = ['#f2c80f', '#fd625e', '#01b8a9']
axs_titles = ['DevSet', 'TestSet']
labels = ['Precision', 'Recall', 'Fscore']
metrics_ner = [[precision_dev[0], recall_dev[0], fscore_dev[0]], [precision_test[0], recall_test[0], fscore_test[0]]]

fig, axs = plt.subplots(1,2, figsize=(16, 10))
title = "VARYING {} - MODEL {}".format(SETTINGS["TEST_TYPE"], SETTINGS["MODEL_NAME"])
fig.suptitle(title, fontsize = 12)
for i in range(len(metrics_ner)):
    axs[i].set_title(axs_titles[i])
    axs[i].set_xlabel(xlabel)
    axs[i].set_ylabel('scores')
    axs[i].grid(True)
    for j in range(len(labels)):
        axs[i].plot(VAR_TEST, metrics_ner[i][j], color = colors[j], label = labels[j])
        axs[i].legend(loc = 1)
fig.savefig(SETTINGS["PLOT_SCORES_PATH"])