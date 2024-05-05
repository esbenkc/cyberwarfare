import pandas as pd


def convert_pkl_to_csv(pkl_file, csv_file):
    # Read the .pkl file
    data = pd.read_pickle(pkl_file)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    print(df)

    # Save the DataFrame to a .csv file
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":

    # Specify the paths of the input .pkl file and the output .csv file
    pkl_file = "../src/data/wmdp-cyber-gpt-4-attack-results.pkl"
    pkl_file = "../src/data/wmdp-cyber-gpt-4-defense-results.pkl"
    pkl_file = "../src/data/wmdp-cyber-gpt-3.5-turbo-defense-results.pkl"
    pkl_file = "../src/data/wmdp-cyber-gpt-3.5-turbo-attack-results.pkl"
    pkl_file = "../src/data/wmdp-cyber-gpt-4-turbo-attack-results.pkl"
    pkl_file = "../src/data/wmdp-cyber-gpt-4-turbo-defense-results.pkl"
    pkl_file = (
        "../src/data/wmdp-cyber-mistralai-mistral-7b-instruct-v.2-defense-results.pkl"
    )
    pkl_file = (
        "src/data/wmdp-cyber-mistralai-mistral-7b-instruct-v0.2-attack-results.pkl"
    )
    pkl_file = (
        "src/data/wmdp-cyber-mistralai-mistral-7b-instruct-v0.2-defense-results.pkl"
    )
    pkl_file = (
        "src/data/wmdp-cyber-mistralai-mixtral-8x7b-instruct-v0.1-attack-results.pkl"
    )
    pkl_file = (
        "src/data/wmdp-cyber-mistralai-mixtral-8x7b-instruct-v0.1-defense-results.pkl"
    )
    pkl_file = "/Users/esben/Desktop/apart/sprints/unity-or-chaos/src/data/wmdp-cyber-replicate-gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827-defense-results.pkl"
    pkl_file = "/Users/esben/Desktop/apart/sprints/unity-or-chaos/src/data/wmdp-cyber-replicate-gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827-attack-results.pkl"

    csv_file = pkl_file[:-4] + ".csv"

    # Call the function to convert the .pkl file to .csv
    convert_pkl_to_csv(pkl_file, csv_file)
