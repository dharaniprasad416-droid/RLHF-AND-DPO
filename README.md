import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# -------------------------------------------------
# 1. Generate candidate answers and gold rewards
# -------------------------------------------------

N = 500
features = rng.normal(size=(N, 3))

# Features represent answer quality, relevance and clarity
gold_weights = np.array([1.0, 0.8, 0.6])
gold_rewards = features @ gold_weights

# Split data into training and testing sets
train_idx = np.arange(0, 350)
test_idx = np.arange(350, N)

# -------------------------------------------------
# 2. Generate preference pairs with annotator noise
# -------------------------------------------------

def make_preferences(indices, noise, n_pairs=1500):
    i = rng.choice(indices, n_pairs)
    j = rng.choice(indices, n_pairs)

    # Avoid identical answers
    valid = i != j
    i, j = i[valid], j[valid]

    # True preferred answer according to gold reward
    prefer_i = gold_rewards[i] > gold_rewards[j]

    # Annotator sometimes chooses the worse answer
    flip = rng.random(len(i)) < noise
    prefer_i = np.logical_xor(prefer_i, flip)

    winner = np.where(prefer_i, i, j)
    loser = np.where(prefer_i, j, i)

    return winner, loser


# -------------------------------------------------
# 3. Train a simple reward model (RLHF component)
# -------------------------------------------------

def train_reward_model(winner, loser, epochs=500, lr=0.03):
    weights = np.zeros(features.shape[1])

    for _ in range(epochs):
        difference = features[winner] - features[loser]
        logits = np.clip(difference @ weights, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))

        gradient = difference.T @ (probabilities - 1) / len(winner)
        weights -= lr * gradient

    return weights


# -------------------------------------------------
# 4. Train a DPO-style linear preference model
# -------------------------------------------------

def train_dpo(winner, loser, epochs=500, lr=0.03, beta=0.5):
    weights = np.zeros(features.shape[1])

    for _ in range(epochs):
        difference = features[winner] - features[loser]
        logits = np.clip(beta * (difference @ weights), -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))

        gradient = beta * (
            difference.T @ (probabilities - 1) / len(winner)
        )
        weights -= lr * gradient

    return weights


# -------------------------------------------------
# 5. Evaluate models on held-out answers
# -------------------------------------------------

def evaluate(weights):
    predictions = features[test_idx] @ weights
    actual = gold_rewards[test_idx]

    # Correlation with the true reward
    correlation = np.corrcoef(predictions, actual)[0, 1]

    # Average gold reward among top 20% predicted answers
    count = max(1, len(predictions) // 5)
    top_indices = np.argsort(predictions)[-count:]
    top_gold = np.mean(actual[top_indices])

    return correlation, top_gold


# -------------------------------------------------
# 6. Compare at three levels of annotator noise
# -------------------------------------------------

noise_levels = [0.0, 0.1, 0.3]
results = {"RLHF reward model": [], "DPO": []}

print("RLHF vs DPO: Annotator Noise Experiment")
print("-" * 65)

for noise in noise_levels:
    winner, loser = make_preferences(train_idx, noise)

    rlhf_weights = train_reward_model(winner, loser)
    dpo_weights = train_dpo(winner, loser)

    rlhf_corr, rlhf_gold = evaluate(rlhf_weights)
    dpo_corr, dpo_gold = evaluate(dpo_weights)

    results["RLHF reward model"].append(rlhf_gold)
    results["DPO"].append(dpo_gold)

    print(f"\nAnnotator noise: {noise:.0%}")
    print(f"RLHF-style model: correlation={rlhf_corr:.3f}, "
          f"top-group gold reward={rlhf_gold:.3f}")
    print(f"DPO-style model:  correlation={dpo_corr:.3f}, "
          f"top-group gold reward={dpo_gold:.3f}")


# -------------------------------------------------
# 7. Plot the comparison
# -------------------------------------------------

x = np.arange(len(noise_levels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(
    x - width / 2, results["RLHF reward model"],
    width, label="RLHF reward model"
)
plt.bar(
    x + width / 2, results["DPO"],
    width, label="DPO"
)

plt.xticks(x, ["0% noise", "10% noise", "30% noise"])
plt.xlabel("Annotator noise")
plt.ylabel("Mean gold reward of top predicted answers")
plt.title("RLHF vs DPO Under Annotator Noise")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
