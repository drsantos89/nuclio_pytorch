"""Run QA app."""
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from nucliopytorch.load import get_data

tqdm.tqdm.pandas()


def main() -> None:
    """Run main function."""
    # download and load model usign sentence-transformers
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # if data is already processed, load it
    if Path("./data/wiki.pkl").exists():
        data = pd.read_pickle("./data/wiki.pkl")
    else:
        data = get_data.get_wikipedia()
        # embded each text
        data["embeddings"] = data["text"].progress_apply(lambda x: model.encode(x))
        # save data
        data.to_pickle("./data/wiki.pkl")

    # main loop
    while True:
        # get question from user
        question = input("Ask a question: ")
        # if user wants to exit, break
        if question == "exit":
            break

        # get embedding of question
        question_embedding = model.encode(question)

        # calculate similarity between question and each text
        data["score"] = data["embeddings"].progress_apply(
            lambda x, question_embedding=question_embedding: np.dot(
                x, question_embedding
            )
        )

        # get text with highest similarity
        context = data.sort_values("score", ascending=False).iloc[0]["text"]

        # QA pipeline
        qa_pipeline = pipeline(
            "question-answering",
            tokenizer="deepset/roberta-base-squad2",
            model="deepset/roberta-base-squad2",
        )

        # get answer
        answer = qa_pipeline(question=question, context=context)

        # print answer
        print("Answer:\n", answer["answer"], "\nScore:\n", answer["score"])


if __name__ == "__main__":
    main()
