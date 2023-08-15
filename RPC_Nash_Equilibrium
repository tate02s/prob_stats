"""    Counter Factual Regret Minimization with Rock Paper Sissors"""

"""
In the begining, choose one of the possible actions with equal probability and have the opponent do the same.
Update the action weights (how frequently an action is chosen) based on the outcome of that round for both players
"""

#      This is similar to the 1 arm bandit Bayesian stratigy in the book "Think Bayes"

class RPSTrainer:
    def __init__(self):
       self.NUM_ACTIONS = 3
       self.possible_actions = np.arange(self.NUM_ACTIONS)
       #      Utility columns  are as follows: rock, paper, sissors. rows are rock 
       self.actionUtility = np.array([
           [0, -1, 1],
           [1, 0, -1],
           [-1, 1, 0]
       ])
       self.regretSum = np.zeros(self.NUM_ACTIONS)
       self.strategySum = np.zeros(self.NUM_ACTIONS)

       self.oppregretSum = np.zeros(self.NUM_ACTIONS)
       self.oppstrategySum = np.zeros(self.NUM_ACTIONS)


    def get_strategy(self, regret_sum):
        
       regret_sum[regret_sum < 0] = 0
       normalizing_sum = sum(regret_sum)
       strategy = regret_sum
       for a in range(self.NUM_ACTIONS):
          if normalizing_sum > 0:
             strategy[a] /= normalizing_sum
          else:
             strategy[a] = 1.0 / self.NUM_ACTIONS

       return strategy
    
    def getAverageStrategy(self, strategySum):
       """Take the weights of each strategy (their playing freq) and normalize them."""
       
       average_strategy = [0, 0, 0]
       normalizing_sum = sum(strategySum)
       for a in range(self.NUM_ACTIONS):
          if normalizing_sum > 0:
             average_strategy[a] = strategySum[a] / normalizing_sum
          else:
             average_strategy[a] = 1.0 / self.NUM_ACTIONS

       return average_strategy
    
    def get_action(self, strategy):
       return np.random.choice(self.possible_actions, p=strategy)
    
    def get_reward(self, myAction, opponentAction):
       return self.actionUtility[myAction, opponentAction]
    
    def train(self, iterations):

       for _ in range(iterations):
          strategy = self.get_strategy(self.regretSum)
          oppStrategy = self.get_strategy(self.oppstrategySum)
          self.strategySum += strategy

          opponent_action = self.get_action(oppStrategy)
          my_action = self.get_action(strategy)
       
          my_reward = self.get_reward(my_action, opponent_action)
          opp_reward = self.get_reward(opponent_action, my_action)

          for a in range(self.NUM_ACTIONS):
             my_regret = self.get_reward(a, opponent_action) - my_reward
             opp_regret = self.get_reward(a, my_action) - opp_reward
             self.regretSum[a] += my_regret
             self.oppregretSum[a] += opp_regret


trainer = RPSTrainer()
trainer.train(10_000)
target_policy = trainer.getAverageStrategy(trainer.strategySum)
opp_target_policy = trainer.getAverageStrategy(trainer.oppstrategySum)
print("Target policy: %s" % (target_policy))       
