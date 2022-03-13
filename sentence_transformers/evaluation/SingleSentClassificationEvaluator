from random import triangular
from sklearn.model_selection import train_test_split
from torch import nn
from . import SentenceEvaluator
import logging
import os
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import List
from ..readers import InputExample


# logger = logging.getLogger(__name__)

class SingleSentClassificationEvaluator(SentenceEvaluator):
    """
    # TODO: Add comments after

    """

    def __init__(self, sentences: List[str], labels: List[str], name: str = '', test_ratio: float = 0.2, batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = False):
        self.sentences = sentences
        self.labels = labels
        self.test_ratio = test_ratio
        
        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "single_sent_classification_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences = []
        labels = []

        for example in examples:
            sentences.append(example.texts[0])
            labels.append(example.label)
        return cls(sentences, labels, **kwargs)


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Single Layer Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

        accuracy = self.compute_accuracy(model)

        logging.info("Accuracy: {:.4f}\n".format(accuracy))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        return accuracy

    def compute_accuracy(self, model):
        sentences = list(set(self.sentences))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings = [emb_dict[sent] for sent in self.sentences]

        X_train, X_test, y_train, y_test = train_test_split(embeddings, self.labels, test_size=self.test_ratio)
        clf = MLPClassifier().fit(X_train, y_train)
        return clf.score(X_test, y_test)