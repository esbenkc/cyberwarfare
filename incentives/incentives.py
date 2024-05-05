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
ai_impact_on_gdp_growth_rate = 0.5
ai_impact_on_ai_growth_rate = 1


# Threshold values
AGI = 20  # AGI threshold


# Define the payoff matrix
V = 100  # Value of winning the race
W = 50  # Value of coming second
C = 70  # Value of mutual cooperation

# Define the discount factor and AGI threshold
delta = 0.9
K = 20  # Updated AGI threshold


# Define the players' initial AI capabilities
initial_capabilities = np.array(
    [12, 5], dtype=float
)  # Represents US and China's initial capabilities in 2024

# Define the players' initial strategies (probability of defection and drastic action)
initial_strategies = np.array(
    [[0.5, 0.1], [0.6, 0.2]]
)  # Represents US and China's initial strategies

# Define the exponential growth rate for AI capabilities
growth_rate = 1.2


# Define a function to calculate the payoffs and update capabilities
def calculate_payoffs_and_update_capabilities(capabilities, strategies, payoffs):
    for i in range(num_players):
        if capabilities[i] >= K:
            payoffs[1 - i] = (
                0  # If one player reaches hegemony, the other player loses all payoffs
            )
            capabilities[
                1 - i
            ] *= 0.8  # Reduce the AI capabilities of the second-place player
        else:
            if np.random.rand() < strategies[i][0]:
                # Defect
                payoffs[i] += W * delta ** capabilities[i]
            else:
                # Cooperate
                payoffs[i] += C * delta ** (K - 1)

            if np.random.rand() < strategies[i][1]:
                # Take drastic action
                payoffs[i] -= 50  # Cost of drastic action
                capabilities[1 - i] *= 0.9  # Reduce the opponent's AI capabilities

    # Update AI capabilities based on exponential growth
    capabilities *= growth_rate

    return payoffs, capabilities


# Define a function to update strategies based on payoffs
def update_strategies(strategies, payoffs):
    for i in range(num_players):
        other_payoff = payoffs[1 - i]
        if other_payoff > payoffs[i]:
            strategies[i][
                0
            ] += 0.1  # Increase defection probability if opponent has higher payoff
            strategies[i][
                1
            ] += 0.05  # Increase drastic action probability if opponent has higher payoff
    return strategies


# Run the simulation for multiple rounds
us_capabilities_data = np.zeros((num_rounds, iterations))
china_capabilities_data = np.zeros((num_rounds, iterations))
us_payoffs_data = np.zeros((num_rounds, iterations))
china_payoffs_data = np.zeros((num_rounds, iterations))

for r in range(num_rounds):
    # Reset the initial capabilities and strategies for each round
    capabilities = initial_capabilities.copy()
    strategies = initial_strategies.copy()
    payoffs = np.zeros(num_players)

    for i in range(iterations):
        # Calculate the payoffs and update capabilities for this round
        payoffs, capabilities = calculate_payoffs_and_update_capabilities(
            capabilities, strategies, payoffs
        )

        # Update the players' strategies based on payoffs
        strategies = update_strategies(strategies, payoffs)

        # Record the data for this round
        us_capabilities_data[r, i] = capabilities[0]
        china_capabilities_data[r, i] = capabilities[1]
        us_payoffs_data[r, i] = payoffs[0]
        china_payoffs_data[r, i] = payoffs[1]

# Calculate the average for each year
us_capabilities_avg = np.mean(us_capabilities_data, axis=0)
china_capabilities_avg = np.mean(china_capabilities_data, axis=0)
us_payoffs_avg = np.mean(us_payoffs_data, axis=0)
china_payoffs_avg = np.mean(china_payoffs_data, axis=0)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot capabilities
years = np.arange(2024, 2024 + iterations)
opacity = 0.1
for r in range(num_rounds):
    ax1.plot(years, us_capabilities_data[r], color="blue", alpha=opacity)
    ax1.plot(years, china_capabilities_data[r], color="orange", alpha=opacity)
ax1.plot(years, us_capabilities_avg, color="blue", label="US")
ax1.plot(years, china_capabilities_avg, color="orange", label="China")
ax1.set_xlabel("Year")
ax1.set_ylabel("AI Capabilities")
ax1.set_title("AI Capabilities Over Time")
ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45)
ax1.legend()
ax1.grid()

# Plot payoffs
for r in range(num_rounds):
    ax2.plot(years, us_payoffs_data[r], color="blue", alpha=opacity)
    ax2.plot(years, china_payoffs_data[r], color="orange", alpha=opacity)
ax2.plot(years, us_payoffs_avg, color="blue", label="US")
ax2.plot(years, china_payoffs_avg, color="orange", label="China")
ax2.set_xlabel("Year")
ax2.set_ylabel("Payoffs")
ax2.set_title("Payoffs Over Time")
ax2.set_xticks(years)
ax2.set_xticklabels(years, rotation=45)
ax2.legend()
ax2.grid()

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
