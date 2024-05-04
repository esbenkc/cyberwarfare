import numpy as np

import matplotlib.pyplot as plt


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


# Example usage
file_path = "/path/to/your/file.npy"
data = read_npy_file(file_path)
visualize_dictionary(data)
