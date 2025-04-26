import torch
import transformers
def fine_tune_model(model, dataset, nomenclatures,nomanclature_tokenizer,save_model = False, epochs=3, batch_size=100, learning_rate=5e-5):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    if save_model:
        model.save_pretrained('./Data/Models/fine_tuned_bert')

def train_tokenizer(nomenclatures, tokenizer, save_model = False):
    """
    Train a tokenizer on the provided nomenclature sequences.
    """
    # Tokenize the nomenclature sequences
    tokenized_nomenclatures = [tokenizer.encode(nomenclature) for nomenclature in nomenclatures]

    # Save the tokenizer
    if save_model:
        tokenizer.save_pretrained('./Data/Models/fine_tuned_tokenizer')
        with open('./Data/Models/fine_tuned_tokenizer.json', 'w') as f:
            f.write(tokenizer.to_json())