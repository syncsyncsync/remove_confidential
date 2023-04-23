import argparse
from remove_confidential import ConfidentialDataset, train
from transformers import AutoTokenizer, DistilBertTokenizer

def main(model_to_use, data_file, num_epochs, train_batch_size, eval_batch_size):
    # Load data from the specified data file
    with open(data_file, "r") as f:
        lines = f.readlines()

    # Remove newline characters
    lines = [line.strip() for line in lines]

    # Create input-output pairs
    data = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

    if model_to_use == "bert":
        model_name = "bert-base-multilingual-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        fine_tuned_model_name = "fine_tuned_multilingual_bert"
    elif model_to_use == "distilbert":
        model_name = "distilbert-base-multilingual-cased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        fine_tuned_model_name = "fine_tuned_multilingual_distilbert"
    else:
        raise ValueError("Invalid model_to_use value. Must be 'bert' or 'distilbert'.")

    # Create the dataset
    dataset = ConfidentialDataset(data, tokenizer)

    # Train the model and save it
    train(
        fine_tuned_model_name=fine_tuned_model_name,
        num_epochs=num_epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )

import argparse

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_to_use",
        type=str,
        default="distilbert",
        choices=["bert", "distilbert"],
        help="Choose between 'bert' and 'distilbert' models.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/sample_data.txt",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    return parser


if __name__ == "__main__":
    args = get_train_parser()
    main(
        args.model_to_use,
        args.data_file,
        args.num_epochs,
        args.train_batch_size,
        args.eval_batch_size,
    )

#
#python train.py --model_to_use bert --data_file data/other_data.txt --num_epochs 5 --train_batch_size 8 --eval_batch_size 8
