"""Bayesian stratigy for the One-Armed-Bandit problem"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empiricaldist import Pmf

slots = [0.1, 0.2, 0.3, 0.4]

xs = np.linspace(0, 1, 101)
prior = Pmf(1, xs)
prior.normalize()

plt.plot(prior)
plt.show()

likelihood = {
    "W": xs,
    "L": 1-xs,
}

def update(pmf, data):
    """Update the probability of winning"""
    pmf *= likelihood[data]
    pmf.normalize()

bandit = prior.copy()

# for outcome in "WLLLLLLLLL":
#     update(bandit, outcome)


#   Here are the actual probabilities for the slot machines:
actual_slot_probs = [0.1, 0.2, 0.3, 0.4]

from collections import Counter 

counter = Counter()
#   Count how many times each machine has been played

def play(i):
    """Play machine i.
    i: index of the machine to play
    returns: string 'W' or 'L'
    """
    counter[i] += 1
    p = actual_slot_probs[i]
    if np.random.random() < p:
        return "W"
    else:
        return "L"

#   For each of the 4 slot machines
# for i in range(4):
#     #   Play it 10 times 
#     for _ in range(10):
#         outcome = play(i)
#         update(beliefs[i], outcome)

def choose(beliefs):
    """Use Thomson sampling to choose a machine.

        Draws a single sample from each distribution.

        Returns: index of the machine that yielded the highest value."""
    
    ps = [b.choice() for b in beliefs]
    
    return np.argmax(ps)

def choose_play_update(beliefs):
    """Chooses a machine, plays it anf updates the prior (beliefs param)"""
     
    machine = choose(beliefs)
    outcome = play(machine)
    update(beliefs[machine], outcome)

beliefs = [prior.copy() for _ in range(4)]

num_plays = 1_000
for _ in range(num_plays):
    choose_play_update(beliefs)

for posterior in beliefs:
    plt.plot(posterior)
    plt.show()

print(counter)
