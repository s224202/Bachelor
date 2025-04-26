import transformers
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import torch
import numpy as np
import sys
from Modules.GlycanDataset import GlycanDataset



def main():
    # Load data
    data = pd.read_csv('./Data/glycan_sequences_full_covers.csv')
    ## limit to 100 sequences for testing
    data = data.head(100)

    encoder


main()