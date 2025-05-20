import torch
import matplotlib.pyplot as plt
from torch import nn
from dataset_actions import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from seaborn import heatmap
import warnings


class HashTable:

    # Create empty bucket list of given size
    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()

    def create_buckets(self):
        return [[] for _ in range(self.size)]

    # Insert values into hash map
    def set_val(self, key, val):

        # Get the index from the key
        # using hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be inserted
            if record_key[0] == key[0] and record_key[1] == key[1] and record_key[2] == key[2]:
                found_key = True
                break

        # If the bucket has same key as the key to be inserted,
        # Update the key value
        # Otherwise append the new key-value pair to the bucket
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    # Return searched value with specific key
    def get_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key being searched
            if record_key[0] == key[0] and record_key[1] == key[1]:
                found_key = True
                break

        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"

    # Remove a value with specific key
    def delete_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be deleted
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

    def __str__(self):
        return "".join(str(item) for item in self.hash_table)


class NN_baseline(nn.Module):

    def __init__(self, dims, device="cpu"):

        self.device = device
        self.mapper = HashTable(np.prod(test_x.shape[:-1]))
        for i in range(len(test_x)):
            for j in range(len(test_x)):
                one_hot = torch.zeros(len(test_x) * len(test_x))
                one_hot[i * len(test_x) + j] = 1
                self.hash_map.set_val(str(test_x[i][j]), one_hot)

        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential()

        for i in range(1, len(dims)):
            self.linear_stack.append(nn.Linear(dims[i-1], dims[i]))
            self.linear_stack.append(nn.ReLU())
        self.linear_stack.append(nn.Linear(dims[-1], 1))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

