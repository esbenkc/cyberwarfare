import pickle as pkl
import matplotlib.pyplot as plt


def load_and_visualize_npy(path):
    # Load the .npy file
    data = pkl.load(path)

    print(data)


# Example usage
if __name__ == "__main__":
    npy_path = "./data/wmdp-cyber.pkl"
    load_and_visualize_npy(npy_path)
