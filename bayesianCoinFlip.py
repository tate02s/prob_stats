import numpy as np
import scipy.stats.distributions as dist
import scipy.stats
import matplotlib.pyplot as plt
from empiricaldist import Pmf

trials = 250
data = 140 * 'H' + 110 * 'T'

def makeBinomialPmf(n, p):
    """Generates a binomial distribution for a the range of succseses, up to n (which is the number if trials), with probability p"""
    # The range of sucsesses, up to n
    ks = np.arange(n+1)
    # Produces the probability of each k in the array ks, for n number of trials, with probability p
    ps = dist.binom.pmf(ks, n, p)

    return Pmf(ps, ks)

# Binomial probability mass function for a discrete random variable (coin in this instance), for 250 trials & probability of sucsess = 0.5
binom_pmf = makeBinomialPmf(trials, 0.5)

# Vizualizes the binomial pmf 
plt.plot(binom_pmf)
plt.show()

rampUp = np.arange(trials//2)
rampDown = np.arange(trials//2, -1, -1)
rampPrior = np.append(rampUp, rampDown)

triPrior = Pmf(rampPrior)
plt.plot(triPrior)
plt.show()

probHeads = np.linspace(0, 1, trials+1)

triLikelihoods = {
    'H': probHeads,
    'T': 1 - probHeads 
}

def bayesUpdate(pmf, dataSet):
    for coinSide in dataSet:
        pmf *= triLikelihoods[coinSide]

    pmf.normalize()

    return pmf

triPriorCopy = triPrior.copy()
posterior = bayesUpdate(triPriorCopy, data)

print(f'Fair coin pmf max probability: {binom_pmf.max_prob()}, Bayesian updated experimental pmf max probability: {posterior.max_prob()}')
plt.plot(posterior)
plt.plot(binom_pmf)
plt.show()

# Now perform a Students T-test to see wether there is strong enough evidence that the coin is in fact biased. Reject the Null Hypothesis (coin sample experiment is from the same distribution as a fair coin) at the 0.05 level
tTest = scipy.stats.ttest_1samp(a=posterior.qs, popmean=binom_pmf.qs.mean())
print(tTest)
