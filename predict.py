import argparse
from remove_confidential import predict

def main(model_to_use, input_text):
    if model_to_use == "bert":
        fine_tuned_model_name = "fine_tuned_multilingual_bert"
    elif model_to_use == "distilbert":
        fine_tuned_model_name = "fine_tuned_multilingual_distilbert"
    else:
        raise ValueError("Invalid model_to_use value. Must be 'bert' or 'distilbert'.")

    predicted_text = predict(input_text, fine_tuned_model_name=fine_tuned_model_name)
    print(f"Input text: {input_text}")
    print(f"Predicted text: {predicted_text}")


def get_predict_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_to_use",
        type=str,
        default="distilbert",
        choices=["bert", "distilbert"],
        help="Choose between 'bert' and 'distilbert' models.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Input text for prediction.",
    )
    return parser

if __name__ == "__main__":
    #args = get_predict_parser().parse_args()
    args = get_predict_parser().parse_args()
    main(args.model_to_use, args.input_text)

