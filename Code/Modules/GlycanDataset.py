from torch.utils.data import Dataset

class GlycanDataset(Dataset):
    def __init__(self, iupac_sequences, smiles_sequences,wurcs_sequences, selfes_sequences, tokenizer, max_length=128):
        self.iupac_sequences = iupac_sequences
        self.smiles_sequences = smiles_sequences
        self.wurcs_sequences = wurcs_sequences
        self.selfes_sequences = selfes_sequences

        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.iupac_sequences)
    
    def __getitem__(self, idx):
        return {
            'iupac': self.iupac_sequences[idx],
            'smiles': self.smiles_sequences[idx],
            'wurcs': self.wurcs_sequences[idx],
            'selfes': self.selfes_sequences[idx]
        }