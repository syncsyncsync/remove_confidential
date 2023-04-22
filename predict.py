import argparse
from remove_confidential import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask confidential information in text')
    parser.add_argument('text', type=str, help='input text to be processed')
    parser.add_argument('--model', type=str, default='fine_tuned_multilingual_bert', help='path to fine-tuned model directory')
    args = parser.parse_args()

    result = predict(args.text, fine_tuned_model_name=args.model)
    print(result)
