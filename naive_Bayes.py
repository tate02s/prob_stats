"""Naive Bayes"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empiricaldist import Pmf
from empiricaldist import Cdf

penguin_data = pd.read_csv("penguins_raw.csv")

def make_cdf_map(df, colname, by="Species"):
    """Make a CDF for each species"""
    cdf_map = {}
    grouped = df.groupby(by)[colname]
    for species, group in grouped:
        cdf_map[species] = Cdf.from_seq(group, name=species)

    return cdf_map

from scipy.stats import norm 

def make_norm_map(df, colname, by="Species"):
    """Make a map from species to norm object"""
    norm_map = {}
    grouped = df.groupby(by)[colname]
    for species, group in grouped:
        mean = group.mean()
        std = group.std()
        norm_map[species] = norm(mean, std)
    
    return norm_map

flipper_map = make_norm_map(penguin_data, "Flipper Length (mm)")

data = 193
hypos = flipper_map.keys()

prior = Pmf(1/3, hypos)

def update_penguin(prior, data, norm_map):
    """Update hypothetical species."""
    hypos = prior.qs
    likelihood = [norm_map[hypo].pdf(data) for hypo in hypos]
    posterior = prior * likelihood
    posterior.normalize()

    return posterior

posterior1 = update_penguin(prior.copy(), 193, flipper_map)

culmen_length_map = make_norm_map(penguin_data, "Culmen Length (mm)")
# posterior2 = update_penguin(prior.copy(), 48, culmen_length_map)

def update_naive(prior, data_seq, norm_maps):
    """Naive Bayesian Classifier
    
    prior: Pmf
    data_seq: sequence of measurements
    norm_maps: sequence of maps from species to distribution

    returns: Pmf representing the posterior distribution
    """

    posterior = prior.copy()
    for data, norm_map in zip(data_seq, norm_maps):
        posterior = update_penguin(posterior, data, norm_map)
    
    return posterior

colnames = ["Flipper Length (mm)", "Culmen Length (mm)"]
norm_maps = [flipper_map, culmen_length_map]

data_seq = 193, 48

penguin_data["Classification"] = np.nan

for i, row in penguin_data.iterrows():
    data_seq = row[colnames]
    posterior = update_naive(prior.copy(), data_seq, norm_maps)
    penguin_data.loc[i, "Classification"] = posterior.max_prob()

def accuracy(df):
    valid = df["Classification"].notna()
    same = df["Species"] == df["Classification"]

    return same.sum() / valid.sum()

print(f"number of valid classifications: {accuracy(penguin_data)}")
