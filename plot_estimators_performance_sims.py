
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors



def generate_group_sizes_uniform_composition_sampler(N, M):
    """
    Uniformly generate cut points across N without replacement. 
    Groups are composed between cut points (and start and end)
    """
    cut_points = np.random.choice(np.arange(1,N), M-1, replace=False)

    cut_points = np.sort(cut_points)
    cut_points = np.concatenate(([0], cut_points, [N]))
    group_sizes = cut_points[1:] - cut_points[:-1]
    
    if 0 in group_sizes:
        print(cut_points)
        print(group_sizes)
        raise ValueError("Generated group sizes contain zero, which is not allowed.")

    return group_sizes

def generate_population(N, group_sizes, sigma_indy, sigma_group):

    """
    Generate a population with given group sizes, individual noise, and group noise.
    Each group has its own bias drawn from a normal distribution.
    """
    population = []
    for size in group_sizes:
        group_bias = np.random.normal(0, sigma_group)
        group_opinions = np.random.normal(group_bias, sigma_indy, size)
        population.append(group_opinions)
    
    return population


def flat_mean(opinions):

    return np.mean(opinions)

def crowd_of_crowds(group_sizes, opinions):

    m = len(group_sizes)

    crowd_means = []

    i = 0
    for n_j in group_sizes:

        crowd_means.append(np.mean(opinions[i:i+n_j]))
        i += n_j

    return np.mean(crowd_means)

def plug_in(group_sizes, opinions):

    m = len(group_sizes)
    n = sum(group_sizes)

    group_means = []
    group_weights = []

    group_opinion_noise = []

    for j in range(m):
        n_j = group_sizes[j]
        group_means.append(np.mean(opinions[:n_j]))
        this_group_opinions_noise = opinions[:n_j] - group_means[j]
        
        opinions = opinions[n_j:]

        group_opinion_noise.extend(this_group_opinions_noise)

    sigma_G = np.std(group_means, ddof=0)
    sigma_I = np.std(group_opinion_noise, ddof=0)

    for j in range(m):
        n_j = group_sizes[j]
        group_weight = 1/(sigma_G**2 + sigma_I**2/n_j)
        group_weights.append(group_weight)

    wococ_optimal_estimated_vars = np.average(group_means, weights=group_weights)

    return wococ_optimal_estimated_vars

def mixed_effects(group_sizes, opinions):

    # Fit a mixed effects model
    import statsmodels.api as sm 
    import statsmodels.formula.api as smf
    n = sum(group_sizes)
    m = len(group_sizes)
    n_js = np.array(group_sizes)
    df = {
        'opinion': opinions,
        'group': np.repeat(np.arange(m), n_js)
    }
    model = smf.mixedlm("opinion ~ 1", df, groups=df["group"])
    result = model.fit()    
    return result.params['Intercept']

def optimal(group_sizes, opinions, sigma_indy, sigma_group):

    m = len(group_sizes)
    n = sum(group_sizes)

    group_means = []
    group_weights = []

    for j in range(m):
        n_j = group_sizes[j]
        group_means.append(np.mean(opinions[:n_j]))
        opinions = opinions[n_j:]

        group_weight = 1/(sigma_group**2 + sigma_indy**2/n_j)

        group_weights.append(group_weight)

    wococ_optimal = np.average(group_means, weights=group_weights)

    return wococ_optimal

def sim_one():

    N = 1000
    M = 8
    sigma_indy = 10
    sigma_group = 1

    group_sizes = generate_group_sizes_uniform_composition_sampler(N, M)

    flat_mean_estimates = []
    coc_estimates = []
    plugin_estimates = []
    me_estimates = []
    optimal_estimates = []

    for n_samples in range(1000):
        # Generate group sizes
        population = generate_population(N, group_sizes, sigma_indy, sigma_group)

        opinions = [opinion for group in population for opinion in group]
        flat_mean_estimate = flat_mean(opinions)
        coc_estimate = crowd_of_crowds(group_sizes, opinions)
        plugin_estimate = plug_in(group_sizes, opinions)
        me_estimate = mixed_effects(group_sizes, opinions)
        optimal_estimate = optimal(group_sizes, opinions, sigma_indy, sigma_group)
        flat_mean_estimates.append(flat_mean_estimate)
        coc_estimates.append(coc_estimate)
        plugin_estimates.append(plugin_estimate)
        me_estimates.append(me_estimate)
        optimal_estimates.append(optimal_estimate)

    # Priont variances
    print("Flat mean variance:", np.var(flat_mean_estimates))
    print("Crowd of crowds variance:", np.var(coc_estimates))
    print("Plug-in variance:", np.var(plugin_estimates))
    print("Mixed effects variance:", np.var(me_estimates))
    print("Optimal variance:", np.var(optimal_estimates))

sim_one()