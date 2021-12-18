
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

MAX_LEN = 350
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#####################################################################################################

# Method to predict clusters from set of messages

# Load the best saved model
model = Model(2)
model.to(device)
model.load_state_dict(torch.load("/saved models/topic_change_best_model_1.pt", map_location=device))

def predict_clusters(messages):
    
    # Initialize cluster details with first message
    predicted_clusters_as_messages = {}
    predicted_clusters_as_numbers = {}
    cluster_number = 1
    predicted_clusters_as_messages[cluster_number] = [messages["line"][0]]
    predicted_clusters_as_numbers[cluster_number] = [0]
    
    # Predict cluster by checking similarity with last message of other clusters
    for i in range(1, len(messages), 1):
        matched_cluster = -1
        
        # Check with last message of all previous clusters
        for j in predicted_clusters_as_messages.keys():
            
            # Check with jth cluster if no relatively recent match is there
            if(matched_cluster == -1 or 
               predicted_clusters_as_numbers[matched_cluster][-1] < predicted_clusters_as_numbers[j][-1]):
                
                # Encode the pair of messages to determine similarity
                encoded_dict = tokenizer.encode_plus(
                                    messages['line'][i].lower(), predicted_clusters_as_messages[j][-1].lower(),
                                    add_special_tokens = True,
                                    max_length = MAX_LEN,
                                    padding='max_length', 
                                    truncation=True,
                                    return_attention_mask = True,
                                    return_tensors = 'pt',
                                   )
                input_ids = encoded_dict['input_ids']
                attention_masks = encoded_dict['attention_mask']
    
                # Predict the similarity with jth cluster
                model.eval()
                with torch.no_grad():
                    input_ids = input_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    outputs = model(input_ids, attention_masks)
                    _, predicted = torch.max(outputs, 1)
            
                # Consider jth cluster is the most recent match
                if(int(predicted) == 1):
                    matched_cluster = j
        
        # If still cluster not found then either it's a self-link or start of a new conversation
        if(matched_cluster != -1):
            predicted_clusters_as_messages[matched_cluster].append(messages['line'][i])
            predicted_clusters_as_numbers[matched_cluster].append(i)
        else:
            cluster_number += 1
            predicted_clusters_as_messages[cluster_number] = [messages["line"][i]]
            predicted_clusters_as_numbers[cluster_number] = [i]
    
    # Return the cluster details 
    return predicted_clusters_as_messages, predicted_clusters_as_numbers

#######################################################################################################

# Method to get ground truth clusters

def get_clusters(cluster_data):
    cluster_details = {}
    for index, row in cluster_data.iterrows():
        if(row["cluster number"] in cluster_details):
            cluster_details[row["cluster number"]].append(index)
        else:
            cluster_details[row["cluster number"]] = [index]
    return cluster_details

#######################################################################################################

# Get both ground truth and predicted clusters for test data

# Change the directory
path = "/data/cluster prediction/test"
os.chdir(path)

all_predicted_clusters_as_messages = {}
all_predicted_clusters_as_numbers = {}
all_ground_truth_clusters_as_numbers = {}

# Iterate through all files to collect the names
for file_name in os.listdir():
    
    # Read cluster data
    cluster_data = pd.read_csv(file_name, index_col=[0])
    print("\nFile name: {}, Shape: {}, Actual number of clusters: {}".format(file_name, cluster_data.shape, cluster_data["cluster number"].max()))
    
    # Get ground truth clusters
    ground_truth_cluster_details = get_clusters(cluster_data)
    all_ground_truth_clusters_as_numbers[file_name[:10]] = ground_truth_cluster_details
    
    # Get predicted clusters
    print("Predicting clusters ...")
    predicted_clusters_as_messages, predicted_clusters_as_numbers = predict_clusters(cluster_data)
    all_predicted_clusters_as_messages[file_name[:10]] = predicted_clusters_as_messages
    all_predicted_clusters_as_numbers[file_name[:10]] = predicted_clusters_as_numbers

# Save all results for future use
with open('/results/all_predicted_clusters_as_messages_1.txt','w') as data: 
      data.write(str(all_predicted_clusters_as_messages))
with open('/results/all_predicted_clusters_as_numbers_1.txt', 'w') as data: 
      data.write(str(all_predicted_clusters_as_numbers))
with open('/results/all_ground_truth_clusters_as_numbers_1.txt','w') as data: 
      data.write(str(all_ground_truth_clusters_as_numbers))

print("\nDone")

#######################################################################################################

# Method to build contingency matrix

def clusters_to_contingency(gt_clusters, predicted_clusters):
    
    contingency_table = {}
    counts_predicted_clusters = {}
    counts_gt_clusters = {}

    # Update contingency table
    for file_name in gt_clusters:
        for p_cluster in predicted_clusters[file_name]:
            current = {}
            contingency_table[file_name + "_pc_" + str(p_cluster)] = current
            for gt_cluster in gt_clusters[file_name]:
                count = len(set(predicted_clusters[file_name][p_cluster]).intersection(set(gt_clusters[file_name][gt_cluster])))
                if count > 0:
                    current[file_name + "_gtc_" + str(gt_cluster)] = count
    
    # Update predicted clusters count(rows)
    for file_name in predicted_clusters:
        for p_cluster in predicted_clusters[file_name]:
            counts_predicted_clusters[file_name + "_pc_" + str(p_cluster)] = len(predicted_clusters[file_name][p_cluster])
    
    # Update ground truth clusters count(columns)
    for file_name in gt_clusters:
        for gt_cluster in gt_clusters[file_name]:
            counts_gt_clusters[file_name + "_gtc_" + str(gt_cluster)] = len(gt_clusters[file_name][gt_cluster])
        
    return contingency_table, counts_predicted_clusters, counts_gt_clusters

###############################################################################################

# Method to calculate Variation of Information(VI)

def calculate_variation_of_information(contingency, row_sums, col_sums):
    total = 0.0
    for row in row_sums:
        total += row_sums[row]

    H_UV = 0.0
    I_UV = 0.0
    for row in contingency:
        for col in contingency[row]:
            num = contingency[row][col]
            H_UV -= (num / total) * math.log(num / total, 2)
            I_UV += (num / total) * math.log(num * total / (row_sums[row] * col_sums[col]), 2)

    H_U = 0.0
    for row in row_sums:
        num = row_sums[row]
        H_U -= (num / total) * math.log(num / total, 2)
    H_V = 0.0
    for col in col_sums:
        num = col_sums[col]
        H_V -= (num / total) * math.log(num / total, 2)

    max_score = math.log(total, 2)
    VI = H_UV - I_UV

    scaled_VI = VI / max_score
    print("\n{:5.2f}   1 - Scaled VI".format(100 - 100 * scaled_VI))

#######################################################################################################

# Build contingency table and find VI

contingency_table, counts_predicted_clusters, counts_gt_clusters = clusters_to_contingency(all_ground_truth_clusters_as_numbers, all_predicted_clusters_as_numbers)
calculate_variation_of_information(contingency_table, counts_predicted_clusters, counts_gt_clusters)

#######################################################################################################

# Calculate One-to-One overlap

from ortools.graph import pywrapgraph

def one_to_one(contingency, row_sums, col_sums):
    row_to_num = {}
    col_to_num = {}
    num_to_row = []
    num_to_col = []
    for row_num, row in enumerate(row_sums):
        row_to_num[row] = row_num
        num_to_row.append(row)
    for col_num, col in enumerate(col_sums):
        col_to_num[col] = col_num
        num_to_col.append(col)

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    start_nodes = []
    end_nodes = []
    capacities = []
    costs = []
    source = len(num_to_row) + len(num_to_col)
    sink = len(num_to_row) + len(num_to_col) + 1
    supplies = []
    tasks = min(len(num_to_row), len(num_to_col))
    for row, row_num in row_to_num.items():
        start_nodes.append(source)
        end_nodes.append(row_num)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    for col, col_num in col_to_num.items():
        start_nodes.append(col_num + len(num_to_row))
        end_nodes.append(sink)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    supplies.append(tasks)
    supplies.append(-tasks)
    for row, row_num in row_to_num.items():
        for col, col_num in col_to_num.items():
            cost = 0
            if col in contingency[row]:
                cost = - contingency[row][col]
            start_nodes.append(row_num)
            end_nodes.append(col_num + len(num_to_row))
            capacities.append(1)
            costs.append(cost)

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])
  
    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow.
    min_cost_flow.Solve()

    # Score.
    total_count = sum(v for _, v in row_sums.items())
    overlap = 0
    for arc in range(min_cost_flow.NumArcs()):
        # Can ignore arcs leading out of source or into sink.
        if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:
            # Arcs in the solution have a flow value of 1. Their start and end nodes
            # give an assignment of worker to task.
            if min_cost_flow.Flow(arc) > 0:
                row_num = min_cost_flow.Tail(arc)
                col_num = min_cost_flow.Head(arc)
                col = num_to_col[col_num - len(num_to_row)]
                row = num_to_row[row_num]
                if col in contingency[row]:
                    overlap += contingency[row][col]
    print("\n{:5.2f}   one-to-one".format(overlap * 100 / total_count))

one_to_one(contingency_table, counts_predicted_clusters, counts_gt_clusters)

##########################################################################################################

# Calculate exact match score(P/R/F)

def exact_match(gold, auto, skip_single=True):
    # P/R/F over complete clusters
    total_gold = 0
    total_matched = 0
    for filename in gold:
        for cluster in gold[filename].values():
            if skip_single and len(cluster) == 1:
                continue
            total_gold += 1
            matched = False
            for ocluster in auto[filename].values():
                if len(set(ocluster).symmetric_difference(set(cluster))) == 0:
                    matched = True
                    break
            if matched:
                total_matched += 1
    match = []
    subsets = []
    supersets = []
    other = []
    prefix = []
    suffix = []
    gap_free = []
    match_counts = []
    subsets_counts = []
    supersets_counts = []
    other_counts = []
    prefix_counts = []
    suffix_counts = []
    gap_free_counts = []
    total_auto = 0
    for filename in auto:
        for cluster in auto[filename].values():
            if skip_single and len(cluster) == 1:
                continue
            total_auto += 1
            most_overlap = 0
            fraction = 0
            count = 0
            is_subset = False
            is_superset = False
            is_prefix = False
            is_suffix = False
            is_gap_free = False
            is_match = False
            for ocluster in gold[filename].values():
                if len(set(ocluster).symmetric_difference(set(cluster))) == 0:
                    is_match = True
                    break

                overlap = len(set(ocluster).intersection(set(cluster)))
                if overlap > most_overlap:
                    most_overlap = overlap
                    gaps = False
                    for v in ocluster:
                        if min(cluster) <= v <= max(cluster):
                            if v not in cluster:
                                gaps = True
                    fraction = 1 - (overlap / len(set(ocluster).union(set(cluster))))
                    count = len(set(ocluster).union(set(cluster))) - overlap

                    is_subset = (overlap == len(cluster))
                    is_superset = (overlap == len(ocluster))
                    if overlap == len(cluster) and (not gaps):
                        is_gap_free = True
                        if min(ocluster) == min(cluster):
                            is_prefix = True
                        if max(ocluster) == max(cluster):
                            is_suffix = True
            if is_match:
                match.append(fraction)
                match_counts.append(count)
            elif is_superset:
                supersets.append(fraction)
                supersets_counts.append(count)
            elif is_subset:
                subsets.append(fraction)
                subsets_counts.append(count)
                if is_prefix:
                    prefix.append(fraction)
                    prefix_counts.append(count)
                elif is_suffix:
                    suffix.append(fraction)
                    suffix_counts.append(count)
                elif is_gap_free:
                    gap_free.append(fraction)
                    gap_free_counts.append(count)
            else:
                other.append(fraction)
                other_counts.append(count)
    print("\nProperty, Proportion, Av Frac, Av Count, Max Count, Min Count")
    if len(match) > 0:
        print("Match        {:5.2f} {:5.2f} {:5.2f}".format(100 * len(match) / total_auto, 100 * sum(match) / len(match), sum(match_counts) / len(match)), max(match_counts), min(match_counts))
    if len(supersets) > 0:
        print("Super        {:5.2f} {:5.2f} {:5.2f}".format(100 * len(supersets) / total_auto, 100 * sum(supersets) / len(supersets), sum(supersets_counts) / len(supersets)), max(supersets_counts), min(supersets_counts))
    if len(subsets) > 0:
        print("Sub          {:5.2f} {:5.2f} {:5.2f}".format(100 * len(subsets) / total_auto, 100 * sum(subsets) / len(subsets), sum(subsets_counts) / len(subsets)), max(subsets_counts), min(subsets_counts))
    if len(prefix) > 0:
        print("Sub-Prefix   {:5.2f} {:5.2f} {:5.2f}".format(100 * len(prefix) / total_auto, 100 * sum(prefix) / len(prefix), sum(prefix_counts) / len(prefix)))
    if len(suffix) > 0:
        print("Sub-Suffix   {:5.2f} {:5.2f} {:5.2f}".format(100 * len(suffix) / total_auto, 100 * sum(suffix) / len(suffix), sum(suffix_counts) / len(suffix)))
    if len(gap_free) > 0:
        print("Sub-GapFree  {:5.2f} {:5.2f} {:5.2f}".format(100 * len(gap_free) / total_auto, 100 * sum(gap_free) / len(gap_free), sum(gap_free_counts) / len(gap_free)))
    if len(other) > 0:
        print("Other        {:5.2f} {:5.2f} {:5.2f}".format(100 * len(other) / total_auto, 100 * sum(other) / len(other), sum(other_counts) / len(other)))

    p, r, f = 0.0, 0.0, 0.0
    if total_auto > 0:
        p = 100 * total_matched / total_auto
    if total_gold > 0:
        r = 100 * total_matched / total_gold
    if total_matched > 0:
        f = 2 * p * r / (p + r)
    print("{:5.2f}   Matched clusters precision".format(p))
    print("{:5.2f}   Matched clusters recall".format(r))
    print("{:5.2f}   Matched clusters f-score".format(f))

exact_match(all_ground_truth_clusters_as_numbers, all_predicted_clusters_as_numbers, skip_single=True)