# BERT (Bidirectional Encoder Representations from Transformers) is the latest research topic in NLP, which essentially
# implements the concept of Transfer Learning to NLP allowing to do complex tasks such as Seq2Seq model, Question-answering and other tasks
# This project makes use of Pytorch to implement this model.

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import warnings
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

warnings.simplefilter(action='ignore', category=FutureWarning)
RANDOM_SEED = 42
MAX_LENGTH = 140
BATCH_SIZE = 32
epochs = 10



np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# BERT base model is pretrained on English using a masked language modelling objective. This model is case-sensitive, so we need to preserve 
# the case of the review while preprocessing the data

PRE_TRAINED_MODEL = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

# Select a CUDA enabled GPU if available, else train on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clean_text(text):
    # remove all unicode characters. In this case = emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text

# The tokenizer basically handles most of the preprocessing of data for the model
class DatasetReview(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_length):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_length = max_length
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_length,
      return_token_type_ids=False,
      pad_to_max_length=True,
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

# data loaders for Pytorch
def create_data_loader(df, tokenizer, max_length, batch_size):
  ds = DatasetReview(
    reviews=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_length=max_length
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

# Fine tuning BERT for better predictions. We add a dropout layer to prevent overfitting by adding a regularization loss and a fully connected output layer
class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()

  cur_loss = []
  true_preds = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    true_preds += torch.sum(preds == targets)
    cur_loss.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return true_preds.double() / n_examples, np.mean(cur_loss)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  cur_loss = []
  true_preds = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      true_preds += torch.sum(preds == targets)
      cur_loss.append(loss.item())

  return true_preds.double() / n_examples, np.mean(cur_loss)

def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values


def create_model(df):

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    print(df_train.shape, df_val.shape, df_test.shape)

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LENGTH, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LENGTH, BATCH_SIZE)

    data = next(iter(train_data_loader))
    print(data.keys())

    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL)
    model = SentimentClassifier(3)
    model = model.to(device)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    F.softmax(model(input_ids, attention_mask), dim=1)
    optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    model_histories = defaultdict(list)
    best_accuracy = 0

    for epoch in range(epochs):

        print('Epoch: {0}/{1}'.format(epoch+1, epochs))
        print('-' * 10)

        training_accuracy, training_loss = train_epoch(model,train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))

        print('Training Loss: {0}  |  Training Accuracy {1}'.format(training_loss, training_accuracy))

        validation_accuracy, validation_loss = eval_model(model,val_data_loader,loss_fn, device, len(df_val))

        print('Validation Loss: {0}  |  Validation Accuracy {1}'.format(validation_loss, validation_accuracy))
        print('\n')

        model_histories['training_accuracy'].append(training_accuracy)
        model_histories['training_loss'].append(training_loss)
        model_histories['validation_accuracy'].append(validation_accuracy)
        model_histories['validation_loss'].append(validation_loss)

        if validation_accuracy > best_accuracy:
            torch.save(model.state_dict(), 'final_model.bin')
            best_accuracy = validation_accuracy

    testing_accuracy, _ = eval_model(model,test_data_loader,loss_fn,device,len(df_test))

    print('Testing Accuracy: {0}'.format(testing_accuracy))
    
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_data_loader)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    # load the data from the CSV file into a dataframe
    df = pd.read_csv('app_reviews.csv')
    df['content'] = df.content.apply(clean_text)

    create_model(df)