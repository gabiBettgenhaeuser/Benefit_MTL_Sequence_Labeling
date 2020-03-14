import numpy
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas import crosstab
from sklearn import metrics
import matplotlib
import conlleval
import torch
import seaborn


class Instance:
    def __init__(self, word, label, left_context, right_context):
        self.word = word
        self.label = label
        self.left_context = left_context
        self.right_context = right_context

    def __str__(self):
        return "input: {},{},{} \nlabels: {}".format(self.left_context, self.word, self.right_context, self.label)


class Corpora:
    def __init__(self, input_separator=" "):
        self.input_separator = input_separator

    def load_conll_dataset(self, path):
        words = []
        pos = []
        chunk = []
        ner = []

        with open(path, mode='r', encoding="utf-8") as file:
            file = numpy.array(list(file))

        for line in file:
            line = line.strip()

            # skip begin of document indicator (used in some files)
            if line.startswith("-DOCSTART-"):
                continue

            # skip empty line / end of sentence marker (used in some files)
            if len(line) == 0:
                continue

            # Skip marker (used in some files)
            if line == "--":
                continue

            parts = line.split(self.input_separator)
            words.append(parts[0])
            pos.append(parts[1])
            chunk.append(parts[2])
            ner.append(parts[3])

        labels = [ner, pos, chunk]
        return words, labels


class Sentence:

    def __init__(self, context_length, padding_word_string="<none>"):
        self.context_length = context_length
        self.padding_word_string = padding_word_string

    def pad_before(self, a_list):
        return (self.context_length - len(a_list)) * [self.padding_word_string] + a_list

    def pad_after(self, a_list):
        return a_list + (self.context_length - len(a_list)) * [self.padding_word_string]

    def input_in_context(self, words, labels):
        instances = []
        for i, word in enumerate(words):
            left_context = words[max(0, i - self.context_length):i]
            right_context = words[i + 1:i + 1 + self.context_length]
            if i in range(0, self.context_length):
                left_context = self.pad_before(left_context)
            if i in range(len(words) - self.context_length, len(words)):
                right_context = self.pad_after(right_context)
            label = [element[i] for element in labels]
            instances.append(Instance(word, label, left_context, right_context))
        return instances


class WordEmbedding:

    def __init__(self, padding_word_string="<none>", padding_word_vector=numpy.zeros(300),
                 unknown_word_vector=numpy.zeros(300)):
        self.padding_word_string = padding_word_string
        self.padding_word_vector = padding_word_vector
        self.unknown_word_vector = unknown_word_vector

    def load_glove(self, glove_path):
        with open(glove_path, mode='r', encoding="utf-8") as f:
            model = {}
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = numpy.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
        self.word_embedding = model

    def load_fasttext(self, fasttext_path):
        path = datapath(fasttext_path)
        self.word_embedding = load_facebook_vectors(path)

    def word_to_embedding(self, word):
        if word == self.padding_word_string:
            return self.padding_word_vector
        elif word in self.word_embedding:
            return self.word_embedding[word]
        else:
            return self.unknown_word_vector

    def instance_to_embedding(self, instance):
        instance.word_emb = self.word_to_embedding(instance.word)
        instance.left_context_emb = [self.word_to_embedding(word) for word in instance.left_context]
        instance.right_context_emb = [self.word_to_embedding(word) for word in instance.right_context]

    def instances_to_embedding(self, instances):
        x_out = []
        for instance in instances:
            self.instance_to_embedding(instance)
            x_temp = []
            x_temp.extend(instance.left_context_emb)
            x_temp.append(instance.word_emb)
            x_temp.extend(instance.right_context_emb)
            x_out.append(x_temp)
        x_out = torch.Tensor(x_out)
        return x_out


class LabelRepresentation:

    def NER_binary(self, label):
        if label == 'O':
            return ('O')
        else:
            return ('ENT')

    def NER_without_prefix(self, label):
        if label.startswith("I-") or label.startswith("B-"):
            return label[2:]
        else:
            return label

    def use_specific_label_map(self, label_to_idx_map):
        self.label_to_idx_map = label_to_idx_map
        self._compute_idx_to_label_map()

    def use_ner_bio_labels(self):
        self.use_specific_label_map(
            {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8})

    def use_synthetic_bio_labels(self):
        self.use_specific_label_map(
            {"O": 0, "B-SYN1": 1, "I-SYN1": 2, "B-SYN2": 3, "I-SYN2": 4, "B-SYN3": 5, "I-SYN3": 6, "B-SYN4": 7,
             "I-SYN4": 8})

    def use_ner_noprefix_labels(self):
        self.use_specific_label_map({"O": 0, "PER": 1, "ORG": 2, "LOC": 3, "MISC": 4})

    def use_synthetic_noprefix_labels(self):
        self.use_specific_label_map({"O": 0, "SYN1": 1, "SYN2": 2, "SYN3": 3, "SYN4": 4})

    def use_ner_binary_labels(self):
        self.use_specific_label_map({"O": 0, "ENT": 1})

    def use_synthetic_binary_labels(self):
        self.use_specific_label_map({"O": 0, "SYN1": 1})

    def _compute_idx_to_label_map(self):
        self.idx_to_label_map = {v: k for k, v in self.label_to_idx_map.items()}

    @staticmethod
    def convert_noprefix_to_bio_labels(old_labels):
        """ Converts a list of IO labels (e.g. ["O", "ORG", "PER", "PER"])
        to BIO labels (["O", "B-ORG", "B-PER", "I-PER"]). IO labels contain
        less information than BIO labels, so adjacent entities might
        be joined (which is however very rare in practice).
        """
        outside_token = "O"

        new_labels = []
        for i, label in enumerate(old_labels):
            if label == outside_token:
                new_labels.append(outside_token)
            else:
                if i > 0 and old_labels[i - 1] == label:
                    new_labels.append("I-" + label)
                else:
                    new_labels.append("B-" + label)
        return new_labels

    def label_to_idx(self, label):
        return self.label_to_idx_map[label]

    def idx_to_label(self, idx):
        return self.idx_to_label_map[idx]

    def labels_instances(self, instances):
        y_out = []
        for instance in instances:
            instance.label_emb = self.label_to_idx_map[instance.label]
            y_out.append(instance.label_emb)
        return torch.Tensor(y_out)

    def labels_to_onehot(self, labels):
        le = LabelEncoder()
        y = le.fit_transform(labels)
        onehot = OneHotEncoder()
        y = onehot.fit_transform(y.reshape(-1, 1)).toarray()
        return list(y)


class Evaluation:

    def __init__(self, separator=" "):
        self.separator = separator

    def create_conlleval_string(self, instances, prediction_labels, idx_label):
        # Expected conlleval script format: "word true_label predicted_label"
        assert len(instances) == len(prediction_labels)
        conlleval_string = ""
        for instance, prediction_label in zip(instances, prediction_labels):
            # only taking NER task
            conlleval_string += "{}{}{}{}{}\n".format(instance.word, self.separator, instance.label[idx_label],
                                                      self.separator,
                                                      prediction_label)
        return conlleval_string

    def evaluate_conlleval_string(self, conlleval_string):
        counts = conlleval.evaluate(conlleval_string.split('\n'), {'delimiter': self.separator})
        full_report = conlleval.report(counts)
        overall, per_label = conlleval.metrics(counts)
        return overall, per_label, full_report

    @staticmethod
    def extract_f_score(evaluation_output):
        """ Extracts from the output given by the CoNLL Perl script
        the value corresponding to the total F1 score.
        """
        line = evaluation_output.split("\n")[1]
        return float(line[-5:])

    def simple_evaluate(self, instances, prediction_labels):
        """ Returns just the f-score (for all NER types)
        Predictions is a label ("MISC", "ORG", etc. not a class vector!)
        """
        conlleval_string = self.create_conlleval_string(instances, prediction_labels)
        eval_output = self.evaluate_conlleval_string(conlleval_string)
        return Evaluation.extract_f_score(eval_output)


class SyntheticData:

    def dim_distinct_labels(self, idx_labels):
        return len(idx_labels)

    def generate_label_matrix(self, idx_labels, alpha):
        dimension = self.dim_distinct_labels(idx_labels)
        matrix = numpy.eye(dimension) * alpha
        matrix[matrix == 0] = (1 - alpha) / (dimension - 1)
        return matrix

    def generate_data(self, true_labels, idx_labels, alpha):
        synthetic_data = []
        label_matrix = self.generate_label_matrix(idx_labels, alpha)
        for given_label in true_labels:
            synthetic_data.append(numpy.random.choice(idx_labels, p=label_matrix[given_label.item()]))
        return torch.tensor(synthetic_data, dtype=torch.long)


class SequenceLabelingDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, n_tokens, task_id):
        self.x = x
        self.y = y
        self.n_tokens = n_tokens
        self.task_id = task_id

    def __len__(self):
        return self.n_tokens

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx, self.task_id])


def predict(model, data_loader, seq_size, input_size, device, num_tasks):
    label_pred = torch.zeros(num_tasks, len(data_loader.dataset))
    label_pred = label_pred.to(device)
    current = 0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.view(-1, seq_size, input_size).requires_grad_()
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 2)
            for x in range(num_tasks):
                label_pred[x, current:current + len(predicted[0])] = predicted[x].float()
            current = current + (len(predicted[0]))
    return label_pred


def MutualInformation(labels, task_number):
    mi = []
    all_combinations = [(i, j) for i in range(task_number) for j in (range(task_number))]
    valid_combinations = sorted(set(tuple(sorted(l)) for l in all_combinations))
    for element in valid_combinations:
        i, j = element
        if i == 0:
            if i != j:
                mi.append(metrics.adjusted_mutual_info_score(labels[:, i], labels[:, j]))
    return mi


import matplotlib.pyplot as plt
import numpy


def LabelCoocurrencePlot(labels, label_type, num_tasks, save_plot_path):
    y_labels = [[] for i in range(num_tasks)]
    lb1 = LabelRepresentation()
    lb2 = LabelRepresentation()

    if label_type == 'BINARY':
        lb1.use_ner_binary_labels()
        lb2.use_synthetic_binary_labels()
    elif label_type == 'IO':
        lb1.use_ner_noprefix_labels()
        lb2.use_synthetic_noprefix_labels()
    elif label_type == 'BIO':
        names = []
        lb1.use_ner_bio_labels()
        lb2.use_synthetic_bio_labels()
    names = list(lb1.idx_to_label_map.values())

    for i in range(num_tasks):
        for idx in range(len(labels)):
            if i == 0:
                y_labels[i].append(lb1.idx_to_label(labels[idx, i].item()))
            else:
                y_labels[i].append(lb2.idx_to_label(labels[idx, i].item()))
    y_labels = numpy.asarray(y_labels)

    for i in range(num_tasks - 1):
        c_prob = crosstab(y_labels[0], y_labels[i + 1], normalize='index', rownames=['NER'],
                          colnames=['AUX {}'.format(i)])
        c_prob = c_prob.reindex(names, axis="rows")

        c_labels = crosstab(y_labels[0], y_labels[i + 1], rownames=['NER'], colnames=['AUX {}'.format(i)])
        c_labels = c_labels.reindex(names, axis="rows")

        matplotlib.use('Agg')

        fig = plt.figure(figsize=(10, 5))
        fig.suptitle("LABELS CO-OCURRENCE", fontsize=13)
        ax1 = plt.subplot(121)
        ax1.set_title("Conditional Probability")
        ax1 = seaborn.heatmap(c_prob, cmap="YlGnBu", annot=True, cbar=False)
        bottom, top = ax1.get_ylim()
        ax1.set_ylim(bottom + 0.5, top - 0.5)

        ax2 = plt.subplot(122)
        ax2.set_title("Label Count")
        axs2 = seaborn.heatmap(c_labels, cmap="YlGnBu", annot=True, fmt='g', cbar=False)
        bottom, top = ax2.get_ylim()
        ax2.set_ylim(bottom + 0.5, top - 0.5)
        fig.savefig(save_plot_path+"_TASK{}.png".format(i))