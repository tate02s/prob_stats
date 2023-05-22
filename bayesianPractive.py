"""Chapter 2 Excercise 1"""
# import pandas as pd
# from fractions import Fraction

# # Solution goes here
# bayesTbl = pd.DataFrame(index=['fair coin', 'unfair coin'])

# #Equal probability of grabbing either the fair or unfair coin
# bayesTbl['prior'] = Fraction(1, 2), Fraction(1, 2)

# #The likelihood of head on the fair coin is 1/2, and the likelihood of heads on unfair coin (double headed coin) is 2/2 = 1
# bayesTbl['likelihood'] = Fraction(1, 2), 1
# #Computing the likelihood of heads, given the fair coin was drawn (the numerator of Bayes Theorem)
# bayesTbl['unnormed posterior'] = bayesTbl['prior'] * bayesTbl['likelihood']

# #Computing the likelihood of heads given all events with the sample space (fair and unfair, in this case)
# bayesTbl['total heads prob'] = sum(bayesTbl['unnormed posterior'])

# #Normalizing the posterior (the left side of the equation for Bayes Theorem)
# bayesTbl['normed posterior'] = bayesTbl['unnormed posterior'] / bayesTbl['total heads prob']

# fairGivenHeads = bayesTbl['normed posterior'].loc['fair coin']

# print(f'P(F|H) = {fairGivenHeads}')
# print(bayesTbl)
#---------------------------------------------------------------------------------------------------------
"""Chapter 2 Excercise 2"""
# import empiricaldist
# import numpy as np
# import pandas as pd
# from fractions import Fraction

# sampleSpace = {'BB': Fraction(1, 4), 'BG': Fraction(1, 4), 'GB': Fraction(1, 4), 'GG': Fraction(1, 4)}

# birthPmf = empiricaldist.Pmf()
# birthPmf.from_seq(])
# print(birthPmf)

#-------------------------------------------------------------------------------------------------------
"""Chapter 2 excercise 3 m&ms"""
import pandas as pd
import numpy as np
from fractions import Fraction

mnm1994 = {'year': '1994', 'brown': 0.3, 'yellow': 0.2, 'red': 0.2, 'green': 0.1, 'orange': 0.1, 'tan': 0.1}
mnm1996 = {'year': '1996', 'blue': 0.24, 'green': 0.2, 'orange': 0.16, 'yellow': 0.14, 'red': 0.13, 'brown': 0.13}
mnms = [mnm1994, mnm1996]

mnm = pd.DataFrame(index=[mnmDict['year'] for mnmDict in mnms], columns=[color for color in set.union(set(mnm1994), set(mnm1996))])

for i, year in enumerate(mnm.index):
    for color in mnm.columns:
       prob = mnms[i][color] if color in mnms[i].keys() else 0
       mnm[color].iloc[i] = prob


bayesTbl = pd.DataFrame(index=['1994 bag', '1996 bag'])

bayesTbl['prior'] = Fraction(1, 2), Fraction(1, 2)



print(mnm)
print(mnm['orange']['1994'])