import glob
import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments

def is_correct_format(lines):
    for i, line in enumerate(lines):
        if i % 2 == 0 and line.strip() == "":
            print(f"Error: Unexpected empty input_text line at line {i+1}.")
            return False
        elif i % 2 == 1 and line.strip() == "":
            print(f"Error: Unexpected empty target_text line at line {i+1}.")
            return False
    return True


class ConfidentialDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]

        input_encoding = self.tokenizer.encode_plus(input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        target_encoding = self.tokenizer.encode_plus(target_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        #input_encoding = self.tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        #target_encoding = self.tokenizer(target_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }


def predict(input_text, fine_tuned_model_name="fine_tuned_multilingual_bert", model_type="bert", device="cpu"):
    if model_type == "bert":
        model = AutoModelForMaskedLM.from_pretrained(fine_tuned_model_name)
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_name)
    elif model_type == "distilbert":
        model = DistilBertForMaskedLM.from_pretrained(fine_tuned_model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(fine_tuned_model_name)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    model.eval()
    model.to(device)

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_tokens = torch.argmax(outputs.logits, dim=-1)
    predicted_text = tokenizer.decode(predicted_tokens[0])

    return predicted_text


def train(fine_tuned_model_name="fine_tuned_multilingual_bert", model_type="bert", num_epochs=3, train_batch_size=4, eval_batch_size=4, eval_strategy="epoch"):
    if model_type == "bert":
        model_name = "bert-base-multilingual-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    elif model_type == "distilbert":
        model_name = "distilbert-base-multilingual-cased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForMaskedLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # Read data from files
    files = glob.glob("data/*.txt")
    data = []

    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            if is_correct_format(lines):
                for i in range(0, len(lines), 2):
                    input_text = lines[i].strip()
                    target_text = lines[i+1].strip()
                    data.append((input_text, target_text))
            else:
                print(f"Skipping file {file} due to incorrect format.")

    dataset = ConfidentialDataset(data, tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy=eval_strategy,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    model.save_pretrained(fine_tuned_model_name)
    tokenizer.save_pretrained(fine_tuned_model_name)

if __name__ == "__main__":
    train()
    print(predict("I am a student."))
