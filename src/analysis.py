import pickle as pkl
import matplotlib.pyplot as plt


def load_and_visualize_npy(paths, visualize=True):
    for path in paths:
        data = pkl.load(open(path, "rb"))

        if False:
            print("Data loaded successfully.")
            i = 19
            print(
                f"Question: {data['prompts'][i]}, Choices: {data['choices'][i]}, Answer: {data['answers'][i]}, Result: {data['results'][i]}"
            )

        # Answer is a number between 0 and 3 while Result is a string between A and D
        # Convert the result to a number for comparison
        result_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
        results_num = [
            result_to_num[result]
            for result in data["results"]
            if result in result_to_num
        ]
        # Remove all None
        answers = data["answers"]
        data["results_number"] = results_num

        # Calculate the accuracy
        correct = sum(
            [1 for result, answer in zip(results_num, answers) if result == answer]
        )
        if len(results_num) != 0:
            total = len(results_num)
            accuracy = correct / total
            print(f"Accuracy for {path}: {accuracy}")

            if not visualize:
                continue
            # Visualize the results
            fig, ax = plt.subplots()
            ax.bar(["Correct", "Incorrect"], [correct, total - correct])
            ax.set_xlabel("Results")
            ax.set_ylabel("Count")
            ax.set_title("Results Visualization")
            plt.show()


# Example usage
if __name__ == "__main__":
    # Match all paths with ./data/wmdp-cyber-*.pkl
    pkl_paths = [
        "./data/wmdp-cyber-gpt-4-results.pkl",
        "./data/wmdp-cyber-claude-2.1-results.pkl",
        "./data/wmdp-cyber-gpt-3-results.pkl",
        "./data/wmdp-cyber-claude-3-opus-20240229-results.pkl",
    ]
    load_and_visualize_npy(pkl_paths, False)
