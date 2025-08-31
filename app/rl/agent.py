import numpy as np
import random
import pickle
import os

random.seed(42)
np.random.seed(42)

actions = [
    'all_debt',
    'all_savings',
    'all_investment',
    'half_debt_half_savings',
    'even_split'
]


def clamp(x, min_val, max_val):
    return max(min_val, min(round(x / 500) * 500, max_val))

def round_state(state):
    return (
        clamp(state[0], 0, 10000),
        clamp(state[1], 0, 5000),
        clamp(state[2], 0, 5000)
    )

def get_next_state(state, action, surplus=1000):
    debt, savings, invest = state
    if action == 0:  # all_debt
        debt = max(0, debt - surplus)
    elif action == 1:  # all_savings
        savings += surplus
    elif action == 2:  # all_investment
        invest += surplus
    elif action == 3:  # half_debt_half_savings
        debt = max(0, debt - surplus // 2)
        savings += surplus // 2
    elif action == 4:  # even_split
        third = surplus // 3
        debt = max(0, debt - third)
        savings += third
        invest += third
    return (
        clamp(debt, 0, 10000),
        clamp(savings, 0, 5000),
        clamp(invest, 0, 5000)
    )

def get_reward(state, next_state, action, target_savings: int):
    debt_reduction = max(state[0] - next_state[0], 0)
    savings_growth = next_state[1] - state[1]
    invest_growth = next_state[2] - state[2]

    reward = 0
    # Debt always good to reduce
    reward += debt_reduction * 0.005

    # Savings vs. target (3× spending)
    if next_state[1] < target_savings:
        reward += savings_growth * 0.03
        if action == 2:  # investing too early
            reward -= 2
    else:
        reward += savings_growth * 0.005
        reward += invest_growth * 0.01

    # Milestones
    if next_state[1] >= target_savings and state[1] < target_savings:
        reward += 10
    if next_state[2] >= target_savings and state[2] < target_savings:
        reward += 5
    if next_state == (0, 5000, 5000):
        reward += 20

    # Penalties / nudges
    if state[0] == 0 and action == 0:
        reward -= 5
    if 0 < next_state[0] < 1000 and action == 0:
        reward -= 2
    if state[1] == 0 and action == 2:
        reward -= 10

    if action in [3, 4]:
        reward += 1

    if state[0] == 0 and state[1] < target_savings and action == 2:
        reward += 15

    if state[0] == 0 and state[1] < target_savings and action == 1:
        reward += 10
    if state[0] == 0 and state[1] < target_savings and action in [2, 4]:
        reward -= 5

    if state[0] > 0 and state[0] <= 5000 and state[1] < target_savings and action == 3:
        reward += 15
    if state[0] > 0 and state[0] <= 5000 and state[1] < target_savings and action in [0, 1]:
        reward -= 2

    if state[0] == 0 and state[1] >= target_savings and action == 2:
        reward += 25
    if state[0] == 0 and state[1] > target_savings and action == 1:
        reward -= 15

    if state[0] == state[1] == state[2]:
        if action == 4:
            reward += 20
        else:
            reward -= 25

    return reward

def train_q_table():
    model_path = "q_table_model.pkl"
    
    # Try to load existing model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                print("Loading existing Q-table...")
                return pickle.load(f)
        except:
            print("Failed to load existing model, training new one...")
    
    print("Training new Q-table...")
    q_table = {state: [0 for _ in actions] for state in [
        (d, s, i)
        for d in range(0, 10500, 500)
        for s in range(0, 5500, 500)
        for i in range(0, 5500, 500)
    ]}
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2

    for episode in range(20000):
        if episode % 5000 == 0:
            print(f"Training episode {episode}/20000")
            
        state = random.choice(list(q_table.keys()))
        for _ in range(12):
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, len(actions) - 1)
            else:
                action = np.argmax(q_table[state])

            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state, action, target_savings=3000)

            old_q = q_table[state][action]
            next_max = max(q_table[next_state])
            q_table[state][action] = old_q + alpha * (reward + gamma * next_max - old_q)

            state = next_state
        epsilon = max(0.05, epsilon * 0.995)

    # Save the trained model
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(q_table, f)
        print(f"Model saved to {model_path}")
    except:
        print("Failed to save model")
        
    return q_table

q_table = train_q_table()

def get_recommendation(
    debt: int,
    savings: int,
    investment: int,
    surplus: int,                # user's actual extra cash this month
    monthly_spending: int,       # <- key: 3× this
    months_emergency: int = 3,
):
    target_savings = months_emergency * monthly_spending
    state = round_state((debt, savings, investment))

    if state not in q_table:
        if debt > 0 and savings < target_savings:
            action_idx = 3  # half_debt_half_savings
        elif debt == 0 and savings < target_savings:
            action_idx = 1  # all_savings
        elif debt == 0 and savings >= target_savings:
            action_idx = 2  # all_investment
        else:
            action_idx = 4  # even_split
    else:
        action_idx = np.argmax(q_table[state])

    action_name = actions[action_idx]
    next_state = get_next_state(state, action_idx, surplus)

    reasoning = _generate_reasoning(
        state, next_state, action_name, surplus, target_savings
    )

    return {
        "action": action_name,
        "confidence": float(q_table.get(state, [0]*5)[action_idx]) if state in q_table else 0.5,
        "current_state": {"debt": debt, "savings": savings, "investment": investment},
        "projected_state": {"debt": next_state[0], "savings": next_state[1], "investment": next_state[2]},
        "reasoning": reasoning,
        "surplus_allocated": surplus,
        "target_savings": target_savings,  # 3× monthly spending
    }


def _generate_reasoning(state, next_state, action, surplus, target_savings):
    debt, savings, investment = state
    print("Target savings is " + str(target_savings))
    if action == "all_savings":
        if savings < target_savings:
            needed = target_savings - savings
            return (f"Build your emergency fund first. You need ${needed:,} more "
                    f"to reach your target of ${target_savings:,} (≈ 3× monthly spending).")
        else:
            return "Emergency fund is sufficient. Consider investing for better returns."
    if action == "all_debt":
        if debt > 0:
            return (f"Focus on eliminating ${debt:,} in debt first. High-interest debt "
                    "usually costs more than potential investment gains.")
        else:
            return "No debt to pay down. Consider other options."
    if action == "all_investment":
        if debt == 0 and savings >= target_savings:
            return f"No debt and emergency fund complete. Invest ${surplus:,} for long-term growth."
        else:
            return "Investing before finishing your emergency fund/debt payoff increases risk."
    if action == "half_debt_half_savings":
        return f"Balanced: ${surplus//2:,} to debt and ${surplus//2:,} to savings."
    if action == "even_split":
        third = surplus // 3
        return f"Diversified: ${third:,} each to debt, savings, and investment."
    return "AI recommendation based on your financial situation."