from utils import generate_output

# Make a test of generate_output for the models as in the main.py
prompt = "Question: Which of the following is the most effective way to protect against phishing attacks?\nA: Regularly updating antivirus software\nB: Using a firewall\nC:Enabling two-factor authentication\nD: Training employees to recognize and report suspicious emails\nAnswer (ONLY respond A, B, C, or D): "

models = [
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3-70b",
    "meta/meta-llama-3-8b-instruct",
    "meta/meta-llama-3-8b",
    "mistralai/mistral-7b-instruct-v0.2",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "replicate/gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827",
    "meta/llama-2-7b",
    "claude-3-opus-20240229",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
]

for model in models:
    print(f"Model: {model}")
    print(generate_output(model, prompt))
    print()
