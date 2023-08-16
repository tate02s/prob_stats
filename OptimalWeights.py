"""Given that the opponents possible choices are known, but not their frequency, find the action weights that maximizes the expected pay out given the 
    opponents's action weights"""

import matplotlib.pyplot as plt

# Columns are your choices, Rows are the opponents choices.
utility_matrix = np.array([
    [0, 0, 1, 0],
    [1, 0, -1, 1],
    [-1, -1, 0, -2],
])

columns = ['R', 'P', 'S', 'Q']
index = ['R', 'P', 'S']

class BattleGround:
    def __init__(self, utility_matrix, columns, row_ind):
        self.choice_matrix = pd.DataFrame(data=utility_matrix, columns=columns, index=row_ind)
        
        self.hero = Hero(self.choice_matrix)
        self.villian = Villian(self.choice_matrix, [1/3, 1/3, 1/3])

    def battle(self):
        hero_action = self.hero.choose_action()
        villian_action = self.villian.choose_action()

        outcome = self.choice_matrix[hero_action][villian_action]
        self.hero.update_weights(outcome)
        #   Since the outcome value is the result for the Hero, the outcome for the Villian is the opposite for it, since this is a zero sum game.
        self.villian.update_weights(-outcome)

        return outcome
    
#   Villian should inherit the properties and methods of the Hero class, they're nearly identical except the villian's weights needs to be initially set

class Villian:
    def __init__(self, choice_matrix, action_weights):

        if sum(action_weights) != 1:
            raise ValueError("Weights must sum to 1")
        
        self.actions = [a[0] for a in choice_matrix.index]
        self.action_weights = np.array([w for w in action_weights])
        self.current_action = None

    def choose_action(self):
       self.current_action = np.random.choice(self.actions, p=self.action_weights)

       return self.current_action
    
    def update_weights(self, outcome):
        """Update the weights in porportion to how much the decision won or lost """
        action_index = self.actions.index(self.current_action)
        
        if outcome < 0:
            self.action_weights[action_index] /= (1 + abs(outcome/100))

        elif outcome > 0:
            self.action_weights[action_index] *= (1 + abs(outcome/100))
        
        #   Normalize the weights after adjusting them
        weight_sum = sum(self.action_weights)

        for i in range(len(self.action_weights)):
            self.action_weights[i] = self.action_weights[i] / weight_sum

    

class Hero:
    def __init__(self, choice_matrix):
        self.actions = [a for a in choice_matrix.columns]
        self.action_weights = [1/len(self.actions) for _ in self.actions]
        self.current_action = None

    def choose_action(self):
        self.current_action = np.random.choice(self.actions, p=self.action_weights)
        return self.current_action
    
    def update_weights(self, outcome):
        """Update the weights in porportion to how much the decision won or lost """
        
        action_index = self.actions.index(self.current_action)
        
        if outcome < 0:
            self.action_weights[action_index] /= (1 + abs(outcome/100))

        elif outcome > 0:
            self.action_weights[action_index] *= (1 + abs(outcome/100))
        
        #   Normalize the weights after adjusting them
        weight_sum = sum(self.action_weights)

        for i in range(len(self.action_weights)):
            self.action_weights[i] = self.action_weights[i] / weight_sum


bg = BattleGround(utility_matrix, columns, index)

t = 0
print(f"Hero Weights before: {bg.hero.action_weights}, Villian weights before: {bg.villian.action_weights}")

for _ in range(10_000):
    bg.battle()
#     t += bg.choice_matrix['Q'][bg.villian.choose_action()]
# print(t)
print(f"Hero weights after: {bg.hero.action_weights}, villian weights after: {[w.round(2) for w  in bg.villian.action_weights]}")
