import numpy as np
import matplotlib.pyplot as plt

# Initial world GDP in trillion USD
initial_gdp = np.array([25.44, 17.96], dtype=float)

# Define the growth rates for the US and China (annual GDP growth rate) from the World Bank 2022
initial_gdp_growth_rate = np.array([0.019, 0.03], dtype=float)
initial_ai_growth_rate = np.array([1.675, 1.413], dtype=float)

# Initial AI capabilities in 2023 (total parameters, source: EpochAI)
initial_ai = np.array([3.4, 1.6], dtype=float)

# Define the number of iterations and players
iterations = 10  # Represents years from 2024 to 2033
num_players = 2
num_rounds = 1

# Define growth rates for AI and GDP
ai_impact_on_gdp_growth_rate = 0.000001
ai_impact_on_ai_growth_rate = 0.01

# Threshold values
AGI = 100  # AGI threshold

# Once AGI is reached, every round the opposing player loses 10% of their AI capabilities to control measures
AGI_impact = 0.5

# Initial strategies
initial_strategies = np.array(
    [[0.5, 0.1], [0.6, 0.2]]
)  # Represents US and China's initial strategies

# An actor can take a drastic action as the other player gets closer to AGI, losing GDP for the expectation of future rewards
# By losing 500 trillion USD in mutual disruption, the player can reduce the other player's AI capabilities by 10% and damage their economy by 250 trillion USD
# This is balanced by how close the other player is to AGI
drastic_action_cost = 0.5
drastic_action_impact = 0.1
drastic_action_gdp_impact = 0.25

# Define the discount factor
delta = 0.9

import numpy as np
import pandas as pd


def update_gdp_and_ai(gdp, ai, strategies, player_index):
    gdp_growth_rate = (
        initial_gdp_growth_rate + strategies[:, 0] + ai_impact_on_gdp_growth_rate * ai
    )
    ai_growth_rate = (
        initial_ai_growth_rate + strategies[:, 1] + ai_impact_on_ai_growth_rate * ai
    )

    gdp *= 1 + gdp_growth_rate
    ai *= ai_growth_rate

    if ai[1 - player_index] >= AGI:
        ai[player_index] *= 1 - AGI_impact
    return gdp, ai


def take_drastic_action(gdp, ai, player_index):
    gdp[player_index] -= drastic_action_cost
    gdp[1 - player_index] -= drastic_action_gdp_impact
    ai[1 - player_index] *= 1 - drastic_action_impact


def run_simulation():
    gdp = initial_gdp.copy()
    ai = initial_ai.copy()
    strategies = initial_strategies.copy()

    results = []

    for year in range(2024, 2034):
        us_action = "None"
        china_action = "None"

        for _ in range(num_rounds):
            for player_index in range(num_players):
                if ai[1 - player_index] >= AGI * 0.8:
                    take_drastic_action(gdp, ai, player_index)
                    if player_index == 0:
                        us_action = "Drastic Action"
                    else:
                        china_action = "Drastic Action"

                gdp, ai = update_gdp_and_ai(gdp, ai, strategies, player_index)

        results.append([year, gdp[0], gdp[1], ai[0], ai[1], us_action, china_action])

    return results


results = run_simulation()

df = pd.DataFrame(
    results,
    columns=[
        "Year",
        "US GDP",
        "China GDP",
        "US AI",
        "China AI",
        "US Action",
        "China Action",
    ],
)
df.to_csv("simulation_results.csv", index=False)

print("Simulation completed. Results saved to 'simulation_results.csv'.")
