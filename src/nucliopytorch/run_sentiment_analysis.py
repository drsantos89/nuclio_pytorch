"""Run sentiment analysis app."""
from transformers import pipeline

MODEL = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"


def main() -> None:
    """Run main function."""
    # use text-classification pipeline for sentiment analysis
    sentiment_classifier = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

    # main loop
    while True:
        # get text from user
        text = input("Enter text: ")
        # if user wants to exit, break
        if text == "exit":
            break

        # print sentiment
        print(sentiment_classifier(text))


if __name__ == "__main__":
    main()
