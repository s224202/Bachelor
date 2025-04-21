import torch
import pandas as pd
import tokenizers

def translate_iupac_to_smiles(iupac_sequence):
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
    tokenizer = tokenizers.Tokenizer.from_file('././Data/Models/fine_tuned_bert_tokenizer.json')
    model = torch.load('././Data/Models/fine_tuned_bert/model.safetensors', weights_only=False)
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

if __name__ == "__main__":
    iupac_sequence = pd.read_csv('././Data/glycan_sequences_full_covers.csv').head(1)['iupac']
    iupac_sequence = iupac_sequence[0]
    smiles_structure = translate_iupac_to_smiles(iupac_sequence)
    print(f"IUPAC: {iupac_sequence}")
    print(f"SMILES: {smiles_structure}")
