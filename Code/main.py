import transformers
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import torch
import numpy as np
import sys
from torch.utils.data import Dataset

class GlycanDataset(Dataset):
    def __init__(self, iupac_sequences, smiles_sequences, tokenizer, max_length=128):
        self.iupac_sequences = iupac_sequences
        self.smiles_sequences = smiles_sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.iupac_sequences)

    def __getitem__(self, idx):
        iupac = self.iupac_sequences[idx]
        smiles = self.smiles_sequences[idx]
        assert idx != None, "Index cannot be None"
        # Tokenize IUPAC and SMILES
        iupac_tokens = self.tokenizer.encode(iupac)
        smiles_tokens = self.tokenizer.encode(smiles)
        # Manually handle padding and truncation
        iupac_tokens.truncate(self.max_length)
        smiles_tokens.truncate(self.max_length)
        iupac_tokens.pad(self.max_length, pad_id=self.tokenizer.token_to_id("[PAD]"))
        smiles_tokens.pad(self.max_length, pad_id=self.tokenizer.token_to_id("[PAD]"))

        return {
            "input_ids": torch.tensor(iupac_tokens.ids, dtype=torch.long),
            "attention_mask": torch.tensor(iupac_tokens.attention_mask, dtype=torch.long),
            "labels": torch.tensor(smiles_tokens.ids, dtype=torch.long),
        }

from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, AdamW

def fine_tune_model(model, tokenizer, iupac_sequences, smiles_sequences, epochs=3, batch_size=100, learning_rate=5e-5):
    """
    Fine-tune the BERT model to map IUPAC sequences to SMILES sequences.
    """
    # Prepare dataset and dataloader
    dataset = GlycanDataset(iupac_sequences, smiles_sequences, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Fine-tuning loop
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save the fine-tuned model
    model.save_pretrained('./Data/Models/fine_tuned_bert')
    tokenizer.save('./Data/Models/fine_tuned_bert_tokenizer.json')

def translate_iupac_to_smiles(model, tokenizer, iupac_sequence):
    """
    Translate an IUPAC sequence into a SMILES structure using the fine-tuned model.
    Args:
        model: The fine-tuned BERT model.
        tokenizer: The tokenizer for decoding SMILES tokens.
        iupac_sequence: The input IUPAC sequence.
    Returns:
        str: The translated SMILES structure.
    """
    # Tokenize the IUPAC sequence
    tokenized_example = tokenizer.encode(iupac_sequence)
    tokenized_tensor = torch.tensor(tokenized_example.ids)

    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(tokenized_tensor.unsqueeze(0))
    
    # Placeholder for decoding logic
    # You need to map the model's output to SMILES token IDs
    for token in outputs:
        token = torch.argmax(token, dim=-1)
    predicted_token_ids = token[0].tolist()
    # Remove padding tokens
    predicted_token_ids = [token for token in predicted_token_ids if token != tokenizer.token_to_id("[PAD]")]

    # Decode the predicted token IDs into a SMILES string
    smiles_structure = tokenizer.decode(predicted_token_ids)
    return smiles_structure

def main():
    # Load data
    data = pd.read_csv('./Data/glycan_sequences_full_covers.csv')
    ## limit to 100 sequences for testing
    data = data.head(100)
    iupac_sequences = data['iupac'].tolist()
    smiles_sequences = data['smiles'].tolist()
    # Tokenize IUPAC and SMILES sequences
    tokenizer = tokenize_nomenclature('iupac_smiles')

    # Load pre-trained BERT model and tokenizer
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tokenizer.get_vocab()))


    # Fine-tune the model
    fine_tune_model(model, tokenizer, iupac_sequences, smiles_sequences)
    # Translate IUPAC to SMILES
    iupac_sequence = data['iupac'].iloc[0]  # Example IUPAC sequence
    smiles_structure = translate_iupac_to_smiles(model, tokenizer, iupac_sequence)
    print(f"Translated SMILES: {smiles_structure}")

def tokenize_nomenclature(nomenclature:str) -> Tokenizer:
    '''
    Tokenize the glycan sequences using the specified nomenclature.
    Args:
        nomenclature (str): The nomenclature to use for tokenization.
    The nomenclature should be one of the following:
        -iupac
        -smiles
        -wurcs-
        -selfies
        -iupac_smiles
    Returns:
        Tokenizer: The trained tokenizer.
    '''
    sequences = pd.read_csv('./Data/glycan_sequences_full_covers.csv').head(100)
    if nomenclature == 'iupac_smiles':
        sequences = sequences[['iupac', 'smiles']]
        sequences = sequences.apply(lambda x: f"{x['iupac']} {x['smiles']}", axis=1)
    else:
        sequences = sequences[nomenclature]
    tokenizer = Tokenizer(BPE())
    pre_tokenizer = ByteLevel()
    tokenizer.pre_tokenizer = pre_tokenizer

    # Train the tokenizer on the glycan sequences
    trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    tokenizer.save(f'./Data/Models/tokenizer_{nomenclature}.json')
    return tokenizer

main()