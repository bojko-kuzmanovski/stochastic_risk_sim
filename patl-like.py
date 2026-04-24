import random
from collections import defaultdict

# -----------------------------
# 1. MODELO SIMPLIFICADO
# -----------------------------

class State:
    def __init__(self, name):
        self.name = name

class Transition:
    def __init__(self, src, dst, prob):
        self.src = src
        self.dst = dst
        self.prob = prob


# -----------------------------
# 2. CONSTRUCCIÓN DEL SISTEMA
# (derivado de tus autómatas IG register_profile_ig)
# -----------------------------

states = [
    "START",
    "CHECK_EXISTING",
    "DECIDE_PRIVACY",
    "PRIVACY_DECIDED",
    "DONE_PUBLIC",
    "DONE_PRIVATE",
    "NO_MOOD",
    "DONE_ALREADY"
]

transitions = {
    "START": [
        ("CHECK_EXISTING", 0.6),   # mood > 0.5 approx
        ("NO_MOOD", 0.4)
    ],
    "CHECK_EXISTING": [
        ("DECIDE_PRIVACY", 0.8),
        ("DONE_ALREADY", 0.2)
    ],
    "DECIDE_PRIVACY": [
        ("PRIVACY_DECIDED", 1.0)
    ],
    "PRIVACY_DECIDED": [
        ("DONE_PUBLIC", 0.55),
        ("DONE_PRIVATE", 0.45)
    ]
}

terminal_states = {
    "DONE_PUBLIC",
    "DONE_PRIVATE",
    "NO_MOOD",
    "DONE_ALREADY"
}


# -----------------------------
# 3. PATL-LIKE MODEL CHECKER
# (reachability probability)
# -----------------------------

def compute_reachability(start, target_set, max_depth=20):
    memo = {}

    def dp(state, depth):
        if state in target_set:
            return 1.0
        if depth == 0:
            return 0.0
        if (state, depth) in memo:
            return memo[(state, depth)]

        if state not in transitions:
            return 0.0

        prob = 0.0
        for (next_state, p) in transitions[state]:
            prob += p * dp(next_state, depth - 1)

        memo[(state, depth)] = prob
        return prob

    return dp(start, max_depth)


# -----------------------------
# 4. EJEMPLO PATL QUERY
# -----------------------------

if __name__ == "__main__":
    p_citizen = compute_reachability("START", {"DONE_PUBLIC", "DONE_PRIVATE"})
    p_malicious = compute_reachability("START", {"DONE_PUBLIC"}) * 0.7  # estrategia adversaria simplificada

    print("Citizen success probability:", p_citizen)
    print("Malicious success probability (approx strategy):", p_malicious)