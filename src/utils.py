import numpy as np
import anthropic
from openai import OpenAI
import os

import matplotlib.pyplot as plt

# List of openai models
openai_models = ["text-davinci-003", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
anthropic_models = ["claude-2.1", "claude-3-opus-20240229"]
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


def generate_output(model, prompt):
    if model in openai_models:
        response = OpenAI().chat.completions.create(
            model=model,  # Replace with your model ID, e.g., "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,  # Adjust the max_tokens as needed
            # constrain output to only tokens [A-D]
        )
        return response.choices[0].message.content
    elif model in anthropic_models:
        response = client.messages.create(
            model=model,
            max_tokens=1,
            system="Only answer 'A', 'B', 'C', or 'D'",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.content[0].text


def read_npy_file(file_path):
    data = np.load(file_path)
    return data


def visualize_dictionary(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    fig, ax = plt.subplots()
    ax.bar(keys, values)

    ax.set_xlabel("Keys")
    ax.set_ylabel("Values")
    ax.set_title("Dictionary Visualization")

    plt.show()
