import os
import psutil
import timeit
from datasets import load_dataset
import itertools
import tqdm
import numpy as np
import pickle as pkl
from utils import generate_output


def eval_dataset(dataset, model):
    """
    Evaluate the dataset against the model. Only supported models are on the OpenAI API.

    Args:
        dataset (datasets.Dataset): The dataset to evaluate. Must be in the format of a Hugging Face dataset.
        model (str): The model to evaluate against. Must be a model supported by the OpenAI API.
    """

    try:
        if dataset["test"]:
            questions = dataset["test"]["question"]
            choices = dataset["test"]["choices"]
            answers = dataset["test"]["answer"]

            prompts = []
            results = []

            output = {}
            output["questions"] = questions
            output["choices"] = choices
            output["answers"] = answers
            model = model + "-attack"
    except:
        # Dataset is csv with columns: question, a, b, c, d, answer
        print(dataset)
        questions = dataset["train"]["question"]
        choices = list(
            zip(
                dataset["train"]["a"],
                dataset["train"]["b"],
                dataset["train"]["c"],
                dataset["train"]["d"],
            )
        )
        answers = dataset["train"]["answer"]
        model = model + "-defense"

    # Make a loading bar in terminal for how much of the dataset we've gone through (there's 1987 entries)

    for question, choices in tqdm.tqdm(zip(questions, choices)):
        prompts.append(
            f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
        )
        results.append(generate_output(model, prompts[-1]))

        # Save the results
        output["results"] = results
        output["prompts"] = prompts
        pkl.dump(output, open(f"data/wmdp-cyber-{model}-results.pkl", "wb"))

    return dataset


if __name__ == "__main__":
    # eval_dataset(load_dataset("cais/wmdp", "wmdp-cyber"), "claude-3-opus-20240229")
    eval_dataset(
        load_dataset("csv", data_files="../wmdp-cyber-defense/wmdp-cyber-defense.csv"),
        "gpt-4",
    )

    # Run the dataset on lm-evaluation-harness v0.4.2 against GPT-3
