from remove_confidential import ConfidentialDataset
from remove_confidential import train
from transformers import DistilBertTokenizer

# Load data from the sample_data.txt file
data_file = "data/sample_data.txt"

with open(data_file, "r") as f:
    lines = f.readlines()

# Remove newline characters
lines = [line.strip() for line in lines]

# Create input-output pairs
data = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

# Load the tokenizer
model_name = "distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Create the dataset
dataset = ConfidentialDataset(data, tokenizer)

# Train the model and save it
train()
