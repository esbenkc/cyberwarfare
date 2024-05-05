import requests
import os
from openai import OpenAI
import pandas as pd


def query_gpt4_api(prompt):
    # Replace 'YOUR_API_KEY' with your actual GPT-4 API key
    api_key = os.environ["OPENAI_API_KEY"]
    prompt = f"What is the quality of this data row?\n\n{prompt}\n\nRespond in this format {{'quality': 1-100, 'incorrect_answer': False/True\}}"
    response = OpenAI().chat.completions.create(
        model="gpt-4",  # Replace with your model ID, e.g., "gpt-4"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,  # Adjust the max_tokens as needed
        # constrain output to only tokens [A-D]
    )
    return response.choices[0].message.content


def filter_quality(df):
    # Filter out the rows with incorrect answers
    return df[df["incorrect_answer"] == False]


if __name__ == "__main__":
    # Read the questions from the wmdp-cyber-defense.csv file
    with open("wmdp-cyber-defense.csv", "r") as file:
        rows = file.readlines()

    # Copy the csv dataset and add two columns for the output of
    # the quality score and the incorrect answer
    df = pd.read_csv("wmdp-cyber-defense.csv")
    df["quality_score"] = None
    df["incorrect_answer"] = None

    # Query the GPT-4 API for each question and print the quality score
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        row = row.strip()
        score = query_gpt4_api(row)
        print(f"Row: \n{row}\n\nQuality Score: {score}\n")
        # Format the quality score and incorrect answer
        quality_score = eval(score)["quality"]
        incorrect_answer = eval(score)["incorrect_answer"]
        # Update the dataframe with the quality score and incorrect answer
        df.at[idx - 1, "quality_score"] = quality_score
        df.at[idx - 1, "incorrect_answer"] = incorrect_answer

    # Save the updated dataframe to a new csv file
    df.to_csv("wmdp-cyber-defense-quality.csv", index=False)
