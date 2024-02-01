import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import time

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import f1_score

class Training_splitted:
    def __init__(self, train_text,
                 train_labels,
                 validation_text,
                 validation_labels,
                 tokenizer,
                 model,
                 batch_size = 32,
                 max_len = 128,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer = 'AdamW',
                 epochs = 2,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=2e-5,
                ):

        self.train_text = train_text
        self.train_labels = train_labels
        self.validation_text = validation_text
        self.validation_labels = validation_labels
        self.epochs = epochs
        self.batch = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.criterion = criterion
        self.lr = lr
        
        if optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def define_input(self, train_text, train_labels, val_text, val_labels):
        sentences_train = [str(sentence) for sentence in train_text]
        sentences_val = [str(sentence) for sentence in val_text]

        tokenized_train_texts = [self.tokenizer.tokenize(sent) for sent in sentences_train]
        tokenized_val_texts = [self.tokenizer.tokenize(sent) for sent in sentences_val]

        train_inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_train_texts],
                                     maxlen=self.max_len, dtype="long", truncating="post", padding="post")

        val_inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_val_texts],
                                   maxlen=self.max_len, dtype="long", truncating="post", padding="post")

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(val_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(val_labels)

        train_data = TensorDataset(train_inputs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch)

        validation_data = TensorDataset(validation_inputs, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch)

        return train_dataloader, validation_dataloader

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self):
        train_dataloader, validation_dataloader = self.define_input(self.train_text, self.train_labels,
                                                              self.validation_text, self.validation_labels)

        for _ in range(self.epochs):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            acc_flag = 0

            self.model.train()

            for batch in tqdm(train_dataloader):
                batch = tuple(t.cuda() for t in batch)
                b_input_ids,  b_labels = batch

                self.optimizer.zero_grad()

                logits = self.model(b_input_ids).logits
                loss = self.criterion(logits, b_labels)

                loss.backward()
                self.optimizer.step()
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                logits = logits.detach().cpu().numpy()
                b_labels = b_labels.detach().cpu().numpy()
                accuracy = self.flat_accuracy(logits, b_labels)
                acc_flag+=accuracy

            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            print("Train acc: {}".format(acc_flag/nb_tr_steps))
            time.sleep(1)

            self.model.eval()

            for batch in validation_dataloader:
                batch = tuple(t.cuda() for t in batch)
                b_input_ids, b_labels = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids).logits

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            time.sleep(1)

class Training_to_split:
    def __init__(self, train_text,
                 train_labels,
                 tokenizer,
                 model,
                 batch_size = 32,
                 max_len = 128,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer = 'AdamW',
                 epochs = 2,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=2e-5,
                ):

        self.train_text = train_text
        self.train_labels = train_labels
        self.epochs = epochs
        self.batch = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.criterion = criterion
        self.lr = lr
        
        if optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def define_input(self, train_text, labels):
        sentences = [str(sentence) for sentence in train_text]
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        
        train_pad = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                     maxlen=self.max_len, dtype="long", truncating="post", padding="post")
    
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_pad, labels, 
                                                                                            random_state=1024,
                                                                                            test_size=0.2)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_data = TensorDataset(train_inputs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch)

        validation_data = TensorDataset(validation_inputs, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch)

        return train_dataloader, validation_dataloader

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self):
        train_dataloader, validation_dataloader = self.define_input(self.train_text, self.train_labels)

        for _ in range(self.epochs):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            acc_flag = 0

            self.model.train()

            for batch in tqdm(train_dataloader):
                batch = tuple(t.cuda() for t in batch)
                b_input_ids,  b_labels = batch

                self.optimizer.zero_grad()

                logits = self.model(b_input_ids).logits
                loss = self.criterion(logits, b_labels)

                loss.backward()
                self.optimizer.step()
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                logits = logits.detach().cpu().numpy()
                b_labels = b_labels.detach().cpu().numpy()
                accuracy = self.flat_accuracy(logits, b_labels)
                acc_flag+=accuracy

            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            print("Train acc: {}".format(acc_flag/nb_tr_steps))
            time.sleep(1)

            self.model.eval()

            for batch in validation_dataloader:
                batch = tuple(t.cuda() for t in batch)
                b_input_ids, b_labels = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids).logits

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            time.sleep(1)
            

class Testing:
    def __init__(self,
                 text,
                 labels,
                 tokenizer,
                 model,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 batch_size = 32,
                 max_len = 128,
                ):

        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len

    def define_input(self):
        sentences_test = [str(sentence) for sentence in self.text]
        tokenized_texts_test = [self.tokenizer.tokenize(sent) for sent in sentences_test]

        test_inputs = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_test],
                                    maxlen=self.max_len, dtype="long", truncating="post", padding="post")

        test_inputs = torch.tensor(test_inputs)
        test_labels = torch.tensor(self.labels)

        test_data = TensorDataset(test_inputs, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)

        return test_dataloader

    def prediction(self):
        test_dataloader = self.define_input()
        predictions = []
        self.model.eval()

        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_labels = batch

            with torch.no_grad():
                logits = self.model(b_input_ids).logits
            logits = logits.detach().cpu().numpy()

            predictions.append(logits)

            flat_predictions = [item for sublist in predictions for item in sublist]
            flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        return flat_predictions

    def print_metrics(self):
        return metrics.classification_report(self.labels, self.prediction(), digits=3)
