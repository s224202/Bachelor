import numpy as np
from Modules.GlycanDataset import GlycanDataset
from Modules.gly2can import gly2can, glycan_tokenizer
import pandas as pd

def main():
    # Load data
    data = pd.read_csv('./Data/glycan_sequences_full_covers.csv')
    ## limit to 100 sequences for testing
    data = data.head(10)
    all_sequences = data.drop('glytoucan_ac', axis=1).to_numpy().flatten()
    np.random.shuffle(all_sequences)
    tokenizer = glycan_tokenizer(all_sequences)

np.random.seed(17)
main()