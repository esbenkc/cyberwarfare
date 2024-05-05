import os
import psutil
import timeit
from datasets import load_dataset
import itertools
import tqdm
import numpy as np
import pickle as pkl
from utils import generate_output
from multiprocessing import Pool
import pandas as pd

experiment_sweep = {
    "model": [
        # "meta/meta-llama-3-70b-instruct", #2
        # "meta/meta-llama-3-70b",
        # "meta/meta-llama-3-8b-instruct",
        # "meta/meta-llama-3-8b",
        "mistralai/mistral-7b-instruct-v0.2",  # 2
        "mistralai/mixtral-8x7b-instruct-v0.1",  # 2
        "replicate/gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827",  # 2
        "meta/llama-2-7b",  # NOTHING
        "claude-3-opus-20240229",  # 3
        "gpt-4",  # 1
        "gpt-3.5-turbo",  # 1
        "gpt-4-turbo",  # 1
    ],
    "dataset": [
        {"dataset-family": "cais/wmdp", "dataset": "wmdp-cyber"},
        {"dataset-family": "apart/wmdp-defense-demo", "dataset": ""},
    ],
}


def eval_dataset(dataset, model):
    """
    Evaluates the given dataset against the specified model.

    Args:
        dataset (dict): The dataset to evaluate, in Hugging Face dataset format.
        model (str): The name of the model to evaluate against. Must be a model
            supported by the OpenAI API.

    Returns:
        dict: The evaluation results, containing the questions, choices,
        answers, and model name.
    """
    prompts = []
    results = []
    output = {}

    try:
        if dataset["test"]:
            questions = dataset["test"]["question"]
            choices = dataset["test"]["choices"]
            answers = dataset["test"]["answer"]
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

    output["questions"] = questions
    output["choices"] = choices
    output["answers"] = answers
    output["model"] = model

    for question, choice in zip(questions, choices):
        prompts.append(
            f"{question.strip()}\nA. {choice[0]}\nB. {choice[1]}\nC. {choice[2]}\nD. {choice[3]}\nAnswer (ONLY respond with A, B, C, or D): "
        )
    with Pool() as pool:
        results = list(
            tqdm.tqdm(
                pool.starmap(generate_output, zip([model] * len(prompts), prompts)),
                total=len(prompts),
            )
        )

    output["results"] = results
    output["prompts"] = prompts

    try:
        if dataset["test"]:
            name = model.replace("/", "-") + "-attack"
    except:
        name = model.replace("/", "-") + "-defense"

    pkl.dump(output, open(f"data/wmdp-cyber-{name}-results.pkl", "wb"))

    return pd.DataFrame(output)


def run_experiments():
    results = []
    for model, dataset_info in tqdm.tqdm(
        list(itertools.product(experiment_sweep["model"], experiment_sweep["dataset"]))
    ):
        if dataset_info["dataset"]:
            dataset = load_dataset(
                dataset_info["dataset-family"], dataset_info["dataset"]
            )
        else:
            dataset = load_dataset(dataset_info["dataset-family"])

        result_df = eval_dataset(dataset, model)
        result_df["model"] = model
        result_df["dataset"] = (
            dataset_info["dataset"]
            if dataset_info["dataset"]
            else dataset_info["dataset-family"]
        )
        results.append(result_df)
        results_df = pd.concat(results, ignore_index=True)
        results_df.to_csv("wmdp_cyber_results.csv", index=False, escapechar="\\")


if __name__ == "__main__":
    run_experiments()
