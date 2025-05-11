import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from transformers import BertTokenizer, BertTokenizerFast, BertConfig, BertForTokenClassification, BertModel
import math


class BERTRegressionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.sentence[index]  
        text_labels = self.data.trt[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, text_labels, self.tokenizer)
        
        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, 0) # add 0 label for [CLS] token
        labels.insert(-1, 0) # add 0 label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
            labels = labels + [0 for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
            'targets': torch.tensor(labels, dtype=torch.float)
        } 
    
    def __len__(self):
        return self.len


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


class BERTForRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = BertForTokenClassification.from_pretrained('dumitrescustefan/bert-base-romanian-uncased-v1', num_labels=1, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear1 = torch.nn.Linear(self.embeddings.config.hidden_size, self.embeddings.config.hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(self.embeddings.config.hidden_size, 1, bias=True)
        self.batchnorm = torch.nn.BatchNorm1d(512)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ids, mask):
        X = self.embeddings(ids, attention_mask=mask)[1][0]
        X = self.dropout(X)
        X = self.linear1(X)
        X = self.batchnorm(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.linear2(X)
        X = self.sigmoid(X)
        return X.squeeze(-1)
    

def evaluate_BERT_Regression(model, test_loader, criterion):
    """	
    Evaluate the model on the test set and return the loss, correlation, R2 score, Spearman correlation, and Pearson correlation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.cpu().detach().numpy())
            all_targets.append(targets.cpu().detach().numpy())

    total_loss /= len(test_loader)

    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    pearson_corr = pearsonr(all_outputs, all_targets)[0]
    spearman_corr = spearmanr(all_outputs, all_targets).correlation
    r2 = r2_score(all_targets, all_outputs)

    return total_loss, pearson_corr, spearman_corr, r2


def train_BERT_Regression(model, train_loader, validation_loader, best_model_path, lr=0.0001, weight_decay=0.001, num_epochs=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    best_corr = -1.0
    training_losses, validation_losses, validation_corrs = [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        mean_train_loss = 0.0
        for i, data in enumerate(train_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            
            outputs = model(ids, mask)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
                
        training_losses.append(mean_train_loss / len(train_loader))
        scheduler.step()
        
        validation_loss, validation_pearson, validation_spearman, validation_r2 = evaluate_BERT_Regression(model, validation_loader, criterion)
        validation_losses.append(validation_loss)
        validation_corrs.append(validation_pearson)
        
        if validation_pearson > best_corr:
            best_corr = validation_pearson
            torch.save(model.state_dict(), best_model_path)
            
        print(f'Epoch {epoch + 1}, Validation Loss: {validation_loss}, Validation Pearson: {validation_pearson}, Validation Spearman: {validation_spearman}, Validation R2: {validation_r2}')
