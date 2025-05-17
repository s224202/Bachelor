import numpy as np
import sklearn.model_selection
from Modules.gly2can import gly2can, glycan_tokenizer
from Modules.fine_tuning import fine_tune_model
from Modules.accuracies import strict_equality, partial_match
import pandas as pd
import torch
import sklearn 
import itertools
def build_model(orig_nomen, target_nomen):
    # Load data
    data = pd.read_csv('./Data/glycan_sequences_full_covers.csv')
    ## limit to 100 sequences for testing
    #data = data.head(1000)
    all_sequences = data.drop('glytoucan_ac', axis=1).to_numpy().flatten()
    np.random.shuffle(all_sequences)
    tokenizer = glycan_tokenizer(all_sequences)
    training_idx, validation_idx = sklearn.model_selection.train_test_split(range(len(data)), test_size=0.1, random_state=17)
    model = gly2can(orig_nomen=orig_nomen, target_nomen=target_nomen)
   
    fine_tune_model(
        gly_model=model,
        training_data=data.iloc[training_idx],
        tokenizer=tokenizer,
        save_model=False,
        epochs=1,
        batch_size=20,
        learning_rate=5e-5
    )
    print("Model fine-tuning complete.")
    ## test the model

    print(data[model.target_nomen].iloc[-1])
    test_input = tokenizer(data[model.orig_nomen].iloc[-1], return_tensors="pt", truncation=True).input_ids
    test_input = test_input.to(model.model.device)
    output = model.model.generate(
        test_input,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        
    )
    combined_output = []
    for i in range(len(output)):
        combined_output.append(tokenizer.decode(output[i], skip_special_tokens=True))
    output = "".join(combined_output)
    output = output.replace(' ', '')
    print(f"Input: {data[model.orig_nomen].iloc[-1]}")
    print(f"Output: {output}")
    print(f"Target: {data[model.target_nomen].iloc[-1]}")

    # validate accuracy
    model.model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for idx in validation_idx:
            test_input = tokenizer(data[model.orig_nomen].iloc[idx], return_tensors="pt", truncation=True).input_ids
            test_input = test_input.to(model.model.device)
            result = model.model.generate(test_input, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            result = tokenizer.decode(result[0], skip_special_tokens=True)
            result = result.replace(' ', '') ## Moldels leave weird spaces in the output
            if result.lower() == data[model.target_nomen].iloc[idx].lower():
                correct += 1
            total += 1
        accuracy = correct / total
        print(f"Validation accuracy: {accuracy:.2f}")

def evaluate_trained_model(orig_nomen, target_nomen):    
    model = gly2can(orig_nomen=orig_nomen, target_nomen=target_nomen, load_model=True)
    data = pd.read_csv('./Data/glycan_sequences_full_covers.csv')
    model.model.eval()
    validation_idx = sklearn.model_selection.train_test_split(range(len(data)), test_size=0.1, random_state=17)[1]
    all_sequences = data.drop('glytoucan_ac', axis=1).to_numpy().flatten()
    np.random.shuffle(all_sequences)
    tokenizer = glycan_tokenizer(all_sequences)

    with torch.no_grad():
        outputs = []
        for idx in validation_idx:
            output = model.model.generate(
                tokenizer(data[model.orig_nomen].iloc[idx], return_tensors="pt", truncation=True).input_ids.to(model.model.device),
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            combined_output = []
            for i in range(len(output)):
                combined_output.append(tokenizer.decode(output[i], skip_special_tokens=True))
            output = "".join(combined_output)
            output = output.replace(' ', '')
            outputs.append(output)
        print(f"Model {orig_nomen} to {target_nomen})")
        print(f"Strict equality: {strict_equality(outputs, data[model.target_nomen].iloc[validation_idx].to_numpy().flatten())}")
        partial_match_score, individual_scores = partial_match(outputs, data[model.target_nomen].iloc[validation_idx].to_numpy().flatten())
        print(f"Partial match: {partial_match_score}")
        with open('./Data/partial_match_scores.txt', 'a') as f:
            f.write(f"{orig_nomen} to {target_nomen}: {individual_scores}\n")
              



np.random.seed(17)
torch.manual_seed(17)
if torch.cuda.is_available():
    torch.cuda.manual_seed(17)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set max memory allocation, CUDA on the HPC is so terrible
    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.cuda.empty_cache()


def main():
    nomenclatues = ['selfies','smiles', 'wurcs','iupac']
    nomenclature_pairs = itertools.permutations(nomenclatues, 2)
    for nomenclatue in nomenclature_pairs:
        print(f"Evaluating {nomenclatue[0]} to {nomenclatue[1]}...")
        evaluate_trained_model(orig_nomen=nomenclatue[0], target_nomen=nomenclatue[1])

if __name__ == "__main__":
    main()