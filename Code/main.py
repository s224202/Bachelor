import numpy as np
from Modules.gly2can import gly2can, glycan_tokenizer
from Modules.fine_tuning import fine_tune_model
import pandas as pd

def main():
    # Load data
    data = pd.read_csv('./Data/glycan_sequences_full_covers.csv')
    ## limit to 100 sequences for testing
    data = data.head(100)
    all_sequences = data.drop('glytoucan_ac', axis=1).to_numpy().flatten()
    np.random.shuffle(all_sequences)
    tokenizer = glycan_tokenizer(all_sequences)

    model = gly2can(orig_nomen='smiles', target_nomen='iupac')
   
    fine_tune_model(
        gly_model=model,
        training_data=data[0:90],
        tokenizer=tokenizer,
        save_model=True,
        epochs=5,
        batch_size=100,
        learning_rate=5e-5
    )
    print("Model fine-tuning complete.")
    ## test the model

    print(data['iupac'].iloc[99])
    test_input = tokenizer(data['smiles'].iloc[99], return_tensors="pt").input_ids
    result = model.model.generate(test_input, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    result = tokenizer.decode(result[0], skip_special_tokens=True)
    print(result)




np.random.seed(17)
main()