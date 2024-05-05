import numpy as np
import anthropic
from openai import OpenAI
import os
import replicate
import matplotlib.pyplot as plt

# List of openai models
openai_models = ["text-davinci-003", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
anthropic_models = ["claude-2.1", "claude-3-opus-20240229"]
replicate_models = [
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3-70b",
    "meta/meta-llama-3-8b-instruct",
    "meta/meta-llama-3-8b",
    "meta/codellama-34b-instruct",
    "mistralai/mistral-7b-instruct-v0.2",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "replicate/gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827",
    "meta/llama-2-7b",
]
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


def generate_output(model, prompt):
    print(f"Model: {model}")
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
    elif model in replicate_models:
        if "mistral" in model:
            response = replicate.run(
                model, input={"prompt": prompt, "max_new_tokens": 1}
            )
        elif "meta" in model:
            response = replicate.run(model, input={"prompt": prompt})
        elif "replicate" in model:
            response = replicate.run(model, input={"prompt": prompt, "max_length": 1})
        return "".join("".join(response))


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
