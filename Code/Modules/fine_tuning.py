import torch
import transformers
import pandas as pd
from Modules.gly2can import gly2can
def fine_tune_model(gly_model:gly2can, training_data, tokenizer,save_model = False, epochs=3, batch_size=100, learning_rate=5e-5):
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

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    




    