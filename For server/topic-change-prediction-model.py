import torch
import torchtext.vocab as vocab
import random
import math
import time
import argparse
import os
import shutil
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.chdir("/data/topic change prediction")

# Load train, dev and test data, also randomly shuffle those
train_dataset = pd.read_csv("train_dataset.csv", index_col=[0])
train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)

dev_dataset = pd.read_csv("dev_dataset.csv", index_col=[0])
dev_dataset = dev_dataset.sample(frac=1).reset_index(drop=True)

test_dataset = pd.read_csv("test_dataset.csv", index_col=[0])
test_dataset = test_dataset.sample(frac=1).reset_index(drop=True)

print(train_dataset.shape, dev_dataset.shape, test_dataset.shape)

print(train_dataset.columns)
print(dev_dataset.columns)
print(test_dataset.columns)

print(train_dataset.head)
print(dev_dataset.head)
print(test_dataset.head)

##################################################################################################

# Set device

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)      
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#################################################################################################

# Method to encode pair of lines in train, dev and test dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
LABEL_COLUMN = "label"

def prepare_data(df, max_len, batch_size):
    input_ids = []
    attention_masks = []

    for i in df.index:
        # Encode line1 and line2 pair
        encoded_dict = tokenizer.encode_plus(
                            df['line1'][i].lower(), df['line2'][i].lower(),
                            add_special_tokens = True,
                            max_length = max_len,
                            padding='max_length', 
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df[LABEL_COLUMN])
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = batch_size
        )
    return dataloader

#################################################################################################

# BeRT based model for topic/context change prediction(binary classification)

class Model(nn.Module):
    def __init__(self, num_labels):
        super(Model, self).__init__()
        self.encode = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.drop_out = nn.Dropout(0.3)
        self.l1 = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_masks):
        outputs = self.encode(input_ids, attention_masks)
        input1 = torch.mean(outputs[2][-2], dim=1)
        input1 = self.drop_out(input1)
        output1 = self.l1(input1)
        return output1

#################################################################################################

# Method for evaluation and find Loss, Matthews correlation coefficient(MCC), Classification report 
# and Confusion matrix

def evaluate_metrics(dataloader, model):
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            labels = batch[2].to(device)
             
            outputs = model(b_input_ids, b_attn_mask)
            loss = criterion(outputs, labels)
            total_loss = total_loss + loss.item()
            
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            y_true.extend(labels.cpu().numpy().tolist()) 
            y_pred.extend(predicted.cpu().numpy().tolist()) 
            
    avg_loss = total_loss/len(dataloader)
    print("MCC : {}".format(matthews_corrcoef(y_true, y_pred)))
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    return avg_loss

###################################################################################################

# Define maximum length, batch size and prepare dataloaders

num_labels = len(train_dataset[LABEL_COLUMN].unique())
print("Number of labels : {}".format(num_labels))

# Set class weights to handle imbalanced class ratios (if required)
class_weights = torch.ones(num_labels)
print("class weights : {}".format(class_weights))

MAX_LEN = 350
print("Max length : {}".format(MAX_LEN))

batch_size = 32
print("Batch size : {}".format(batch_size))

print("Loading Train data")
train_dataloader = prepare_data(train_dataset, MAX_LEN, batch_size)
print("Loading Test data")
test_dataloader = prepare_data(test_dataset, MAX_LEN, batch_size)
print("Loading Validation data")
valid_dataloader = prepare_data(dev_dataset, MAX_LEN, batch_size)

print("Size of Train loader : {}".format(len(train_dataloader)))
print("Size of Valid loader : {}".format(len(valid_dataloader)))
print("Size of Test loader : {}".format(len(test_dataloader)))

##################################################################################################

# Define criteria, scheduler and optimizer for the model
# Specify other hyperparameters like epoch, learning rate etc.

torch.cuda.empty_cache()

model = Model(num_labels)
model.to(device)

clip = 2.0
num_epoch = 4
best_valid_loss = 9999
best_test_loss = 9999
best_train_loss = 0
best_model = 0
model_copy = type(model)(num_labels)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

total_steps = len(train_dataloader) * num_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

##################################################################################################

print("Starting training ...")

for epoch in range(num_epoch):
    model.train()
    print("Epoch {} --------------------------".format(epoch+1))
    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):
    	if(i%10 == 0):
			print("-----------batch : ", i)
        b_input_ids = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(b_input_ids, b_attn_mask)
        loss = criterion(outputs, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

    print("Training Accuracy :-")
    train_loss = evaluate_metrics(train_dataloader, model)
    print("Validation Accuracy :-")
    valid_loss = evaluate_metrics(valid_dataloader, model)
    print("Test Accuracy :-")
    test_loss = evaluate_metrics(test_dataloader, model)
    print("Epoch {} : Train loss = {} : Valid loss = {} : Test loss = {}".format(epoch + 1, train_loss, valid_loss, test_loss))
    if(valid_loss < best_valid_loss):
        best_valid_loss = valid_loss
        best_test_loss = test_loss
        best_train_loss = train_loss
        best_model = epoch+1
        model_copy.load_state_dict(model.state_dict())
        print("Model {} copied".format(epoch+1))

print('Finished Training ...')

#####################################################################################################

# Save the best model for future use

PATH = os.path.join("saved models" , 'topic_change_best_model_1.pt')
torch.save(model_copy.state_dict(), 'topic_change_best_model_1.pt')
model.to('cpu')
model_copy.to(device)
print("---Best model---")
print("Epoch {} : Train loss = {} : Validation Loss = {} : Test loss = {}".format(best_model, best_train_loss, best_valid_loss, best_test_loss))
print("Training Accuracy :-")
train_loss = evaluate_metrics(train_dataloader, model_copy)
print("Validation Accuracy :-")
valid_loss = evaluate_metrics(valid_dataloader, model_copy)
print("Test Accuracy :-")
test_loss = evaluate_metrics(test_dataloader, model_copy)
print("Verifying Epoch {} : Train loss = {} : Validation Loss = {} : Test loss = {}".format(best_model, train_loss, valid_loss, test_loss))
print("done")
