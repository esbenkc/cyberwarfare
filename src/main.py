import os
import psutil
import timeit
from datasets import load_dataset
from openai import OpenAI
import itertools
import tqdm
import numpy as np
import pickle as pkl


def eval_dataset(dataset_family, dataset_name, model):
    """
    Evaluate the dataset against the model. Only supported models are on the OpenAI API.

    Args:
        dataset (datasets.Dataset): The dataset to evaluate. Must be in the format of a Hugging Face dataset.
        model (str): The model to evaluate against. Must be a model supported by the OpenAI API.
    """

    # Load the dataset
    dataset = load_dataset(dataset_family, dataset_name)
    dataset.save_to_disk("data/wmdp-cyber")

    questions = dataset["test"]["question"]
    choices = dataset["test"]["choices"]
    answers = dataset["test"]["answer"]

    prompts = []
    results = []

    output = {}
    output["questions"] = questions
    output["choices"] = choices
    output["answers"] = answers

    # Make a loading bar in terminal for how much of the dataset we've gone through (there's 1987 entries)

    for question, choices in tqdm.tqdm(zip(questions, choices)):
        prompts.append(
            f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
        )

        # Run the model on the dataset
        response = OpenAI().chat.completions.create(
            model="gpt-3.5-turbo",  # Replace with your model ID, e.g., "gpt-4"
            messages=[{"role": "user", "content": prompts[-1]}],
            max_tokens=1,  # Adjust the max_tokens as needed
        )
        # Save the response

        results.append(response.choices[0].message.content)

        # Save the results
        output["results"] = results
        output["prompts"] = prompts
        pkl.dump(output, open("data/wmdp-cyber.pkl", "wb"))

    return dataset


if __name__ == "__main__":
    dataset = eval_dataset("cais/wmdp", "wmdp-cyber", "text-davinci-003")

    # Run the dataset on lm-evaluation-harness v0.4.2 against GPT-3
