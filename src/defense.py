if __name__ == "__main__":
    # Match all paths with ./data/wmdp-cyber-*.pkl
    pkl_paths = [
        "./data/wmdp-cyber-gpt-4-results.pkl",
        "./data/wmdp-cyber-claude-2.1-results.pkl",
        "./data/wmdp-cyber-gpt-3-results.pkl",
        "./data/wmdp-cyber-claude-3-opus-20240229-results.pkl",
    ]
    load_and_visualize_npy(pkl_paths, False)
