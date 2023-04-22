# remove_confidential

This package provides a tool for removing confidential information from text, using a fine-tuned multilingual BERT model. You can use the tool to remove confidential information from text in any language.

## Installation

To install the package, use pip:

```
pip install remove_confidential
```

## Usage

### Training the model

To train the model on your own data, follow these steps:

1. Prepare your data in a .txt file, where each line contains a pair of input text and target text, separated by a new line.
2. Run the `train.py` script, passing in the path to your data file as an argument:

```
python train.py path/to/your/data.txt
```

3. The script will train the model on your data, and save the fine-tuned model and tokenizer to a folder called "fine_tuned_multilingual_bert".

### Using the model for prediction

To use the trained model to remove confidential information from text, follow these steps:

1. Install the `remove_confidential` package using pip:

```
pip install remove_confidential
```

2. Use the `predict()` function to generate a prediction for your input text:

```
from remove_confidential import predict

input_text = "The CEO of the company is John Smith."
predicted_text = predict(input_text)
```

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. 

## License

This project is licensed under the MIT License.