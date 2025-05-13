import torch
import transformers
import pandas as pd
from Modules.gly2can import gly2can
import sklearn.model_selection
def fine_tune_model(gly_model:gly2can, training_data, tokenizer,save_model = False, epochs=10, batch_size=5, learning_rate=5e-5):
    """
    Fine-tune a pre-trained model on a given dataset.

    Args:
        model: The pre-trained model to fine-tune.
        training_data: The dataset to use for fine-tuning.
        tokenizer: The tokenizer to use for the model.
        save_model: Whether to save the fine-tuned
            model or not.
    """
    model = gly_model.model
    model.train()
    training_data, test_data = sklearn.model_selection.train_test_split(training_data, test_size=0.2, random_state=17)
    # Setup of the training data
    training_inputs = training_data[gly_model.orig_nomen]
    training_labels = training_data[gly_model.target_nomen]
    training_inputs = tokenizer(training_inputs.tolist(), padding=True, truncation=True, return_tensors="pt").input_ids
    training_labels = tokenizer(training_labels.tolist(), padding=True, truncation=True, return_tensors="pt").input_ids

    # Create a DataLoader for the training data
    dataset = torch.utils.data.TensorDataset(training_inputs, training_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    accuracies = []
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            #attention_mask = tokenizer(training_data[gly_model.orig_nomen].tolist(), padding=True, truncation=True, return_tensors="pt").attention_mask
            #attention_mask = attention_mask.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for idx in range(len(test_data)):
                test_input = tokenizer(test_data[gly_model.orig_nomen].iloc[idx], return_tensors="pt", truncation=True).input_ids
                test_input = test_input.to(device)
                result = model.generate(test_input, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                result = tokenizer.decode(result[0], skip_special_tokens=True)
                if result == test_data[gly_model.target_nomen].iloc[idx]:
                    correct += 1
                total += 1
            accuracy = correct / total
            accuracies.append(accuracy)
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    if save_model:
        model.save_pretrained(f'./Models/{gly_model.orig_nomen}_{gly_model.target_nomen}_fine_tuned')