

import numpy as np


def sanity_check():

    n_samples = 140  # number of variables in each run
    n_runs = 10000   # number of simulation runs
    mu = 0          # population mean
    sigma = 2       # population standard deviation

    # Generate n_runs sets of n_samples normal random variables
    # Shape will be (n_runs, n_samples)
    data = np.random.normal(mu, sigma, (n_runs, n_samples))


    # Calculate the mean for each run
    sample_means = np.mean(data, axis=1)

    # Calculate the variance of these means
    empirical_variance = np.var(sample_means)

    # The theoretical variance of the mean is σ²/n
    theoretical_variance = (sigma**2) / n_samples

    print(f"Number of samples per run: {n_samples}")
    print(f"Number of runs: {n_runs}")
    print(f"Population variance (σ²): {sigma**2}")
    print(f"\nTheoretical variance of mean: {theoretical_variance:.6f}")
    print(f"Empirical variance of mean:   {empirical_variance:.6f}")

def sanity_check_2():

    n_samples_per_group = 2
    n_groups = 5
    n_samples = n_samples_per_group * n_groups
    n_runs = 10000000   # number of simulation runs
    mu = 0          # population mean
    sigma_group = 2       # population standard deviation
     
    u_js = np.random.normal(mu, sigma_group, (n_runs, n_samples))

    # Just duplicate the data n_samples_per_group times
    data = np.tile(u_js, n_samples_per_group)

    print(data.shape)

    # Calculate the mean for each run
    sample_means = np.mean(data, axis=1)

    # Calculate the variance of these means
    empirical_variance = np.var(sample_means)

    # The theoretical variance of the mean is σ²/n
    theoretical_variance = (sigma_group**2) / n_samples

    print(f"Number of samples per run: {n_samples}")
    print(f"Number of runs: {n_runs}")
    print(f"Population variance (σ²): {sigma_group**2}")
    print(f"\nTheoretical variance of mean: {theoretical_variance:.6f}")
    print(f"Empirical variance of mean:   {empirical_variance:.6f}")



def generate_opinions(group_sizes=[], sigma_indy=1, sigma_group=1, truth=0, seed=1):
    """
    Model is y_ij = θ + u_j + e_ij
    """
    if seed:
        np.random.seed(seed)

    m = len(group_sizes)
    n = sum(group_sizes)

    u_js = np.random.normal(0, sigma_group, m)
    eijs = np.random.normal(0, sigma_indy, n)

    opinions = []
    i = 0
    for j in range(m):
        n_j = group_sizes[j]
        for ij in range(n_j):
            opinions.append(truth + u_js[j] + eijs[i])
            i += 1
    
    return opinions


def wisdom_of_the_crowd(opinions):

    return np.mean(opinions)


def wisdom_of_the_crowd_of_crowds(group_sizes, opinions):

    m = len(group_sizes)

    crowd_means = []

    i = 0
    for n_j in group_sizes:

        crowd_means.append(np.mean(opinions[i:i+n_j]))
        i += n_j

    return np.mean(crowd_means)

def sim_lots(n_sims):

    wocs = []
    wococs = []

    mean_group_1s = []

    group_sizes = [1,2,1,3,1]*20
    sigma_indy = 1
    sigma_group = 0.3
    truth = 0

    for seed in range(n_sims):
        print(seed)
        opinions = generate_opinions(group_sizes, sigma_indy, sigma_group, truth, seed=None)
        wocs.append(wisdom_of_the_crowd(opinions))
        wococs.append(wisdom_of_the_crowd_of_crowds(group_sizes, opinions))

        mean_group_1s.append(np.mean(opinions[:group_sizes[0]]))

    # PLot a historgram of the two estimators
    import matplotlib.pyplot as plt
    plt.hist(wocs, bins=30, alpha=0.5, label='WOC')
    plt.hist(wococs, bins=30, alpha=0.5, label='WOCOC')
    plt.legend(loc='upper right')

    var_woc = np.var(wocs, ddof=1)
    var_wococ = np.var(wococs, ddof=1)


    print("WOC Empirical simulated variance: ", var_woc)

    print("WOC anaytical variance: ")
    print(get_variance_woc(group_sizes, sigma_indy, sigma_group))
    print(get_variance_woc_new(group_sizes, sigma_indy, sigma_group))
    print(get_variance_woc_new_2(group_sizes, sigma_indy, sigma_group))
    print("S preidtced variance: ")
    print(get_variance_woc_S(group_sizes, sigma_indy, sigma_group))
    print("CV predicted variance: ")
    print(get_variance_woc_cv(group_sizes, sigma_indy, sigma_group))

    print("WOCOC Empirical simulated variance: ", var_wococ)
    print("WOCOC anaytical variance: ")
    print(get_variance_wococ(group_sizes, sigma_indy, sigma_group))
    print(get_variance_wococ_new(group_sizes, sigma_indy, sigma_group))
    print(get_variance_wococ_new_2(group_sizes, sigma_indy, sigma_group))
    print("WOCOC H predicted variance: ")
    print(get_variance_wococ_H(group_sizes, sigma_indy, sigma_group))


    print("var_flat_over_var_coc = ", get_variance_woc(group_sizes, sigma_indy, sigma_group) / 
          get_variance_wococ(group_sizes, sigma_indy, sigma_group))
    print(get_variance_ratio_flat_over_coc(group_sizes, sigma_indy, sigma_group))
    print(get_variance_ratio_coc_over_flat(group_sizes, sigma_indy, sigma_group))

    N = sum(group_sizes)
    M = len(group_sizes)
    D = np.sum([n_j**2 for n_j in group_sizes]) / sum(group_sizes)**2
    H_inv = np.mean([1/n_j for n_j in group_sizes])
    rho = sigma_indy**2 / sigma_group**2

    print("var_flat_over_var_coc compressed = ", get_var_ratio_coc_over_flat_compressed(N, M, D, H_inv, rho))


def get_variance_ratio_flat_over_coc(group_sizes, sigma_indy, sigma_group):

    mean_group_sizes = np.mean(group_sizes)
    mean_squared_group_sizes = np.mean([n_j**2 for n_j in group_sizes])
    mean_inverse_group_sizes = np.mean([1/n_j for n_j in group_sizes])
    
    NUMERATOR = 1/mean_group_sizes + mean_squared_group_sizes / mean_group_sizes**2 * sigma_group**2/ sigma_indy**2
    DENOMINATOR = mean_inverse_group_sizes + sigma_group**2 / sigma_indy**2
    return NUMERATOR / DENOMINATOR
    
def get_variance_ratio_coc_over_flat(group_sizes, sigma_indy, sigma_group):

    mean_group_sizes = np.mean(group_sizes)
    mean_squared_group_sizes = np.mean([n_j**2 for n_j in group_sizes])
    mean_inverse_group_sizes = np.mean([1/n_j for n_j in group_sizes])
    
    NUMERATOR = mean_inverse_group_sizes + sigma_group**2 / sigma_indy**2
    DENOMINATOR = 1/mean_group_sizes + mean_squared_group_sizes / mean_group_sizes**2 * sigma_group**2/ sigma_indy**2
    return NUMERATOR / DENOMINATOR

def get_var_ratio_coc_over_flat_compressed(N, M, D, H_inv, rho):

    A = N/M
    numerator = rho*H_inv + 1
    denominator = rho/A + M*D
    return numerator / denominator

def get_variance_woc(group_sizes, sigma_a, sigma_b):

    n = sum(group_sizes)
    m = len(group_sizes)

    variance_indy_component = 1/n * sigma_a**2
    variance_group_component = 1/n**2 * sigma_b** 2 * sum([n_j**2 for n_j in group_sizes])

    return variance_indy_component + variance_group_component


def get_variance_wococ(group_sizes, sigma_a, sigma_b):

    n = sum(group_sizes)
    m = len(group_sizes)

    variance_indy_component = 1/m**2 * sigma_a**2 * sum([1/n_j for n_j in group_sizes])
    variance_group_component = 1/m * sigma_b**2

    return variance_indy_component + variance_group_component


def get_variance_woc_new(group_sizes, sigma_a, sigma_b):

    n = sum(group_sizes)
    m = len(group_sizes)

    variance_indy_component = 1/n * sigma_a**2
    variance_group_component = 1/m * sigma_b** 2 * np.mean([(m*n_j/n)**2 for n_j in group_sizes])

    return variance_indy_component + variance_group_component

def get_variance_woc_new_2(group_sizes, sigma_a, sigma_b):

    n = sum(group_sizes)
    m = len(group_sizes)

    mean_nj = np.mean(group_sizes)
    mean_nj_squared = np.mean([n_j**2 for n_j in group_sizes])

    variance_indy_component = 1/n * sigma_a**2
    variance_group_component = 1/m * sigma_b** 2 * mean_nj_squared / mean_nj**2

    return variance_indy_component + variance_group_component


def get_variance_woc_S(group_sizes, sigma_I, sigma_G):

    n = sum(group_sizes)
    m = len(group_sizes)

    sum_nj_squared = sum([n_j**2 for n_j in group_sizes])

    S = sum_nj_squared/n**2 - 1/m


    variance_indy_component = 1/n * sigma_I**2
    variance_group_component = S* sigma_G** 2 + (1/m) * sigma_G** 2

    return variance_indy_component + variance_group_component

def get_variance_woc_cv(group_sizes, sigma_I, sigma_G):

    n = sum(group_sizes)
    m = len(group_sizes)

    var_nj = np.var(group_sizes, ddof=0)

    S = m/n**2 * var_nj

    cv_nj = np.sqrt(var_nj) / (n/m)

    variance_indy_component = 1/n * sigma_I**2
    variance_group_component = (1/m) * sigma_G** 2 + 1/m * cv_nj**2 * sigma_G** 2

    return variance_indy_component + variance_group_component

    
def get_variance_wococ_new(group_sizes, sigma_a, sigma_b):

    n = sum(group_sizes)
    m = len(group_sizes)

    variance_indy_component = 1/n * sigma_a**2 * np.mean([n/(m*n_j) for n_j in group_sizes])
    variance_group_component = 1/m * sigma_b** 2

    return variance_indy_component + variance_group_component


def get_variance_wococ_new_2(group_sizes, sigma_a, sigma_b):

    n = sum(group_sizes)
    m = len(group_sizes)

    mean_nj = np.mean(group_sizes)
    mean_nj_inverse = np.mean([1/n_j for n_j in group_sizes])

    variance_indy_component = 1/n * sigma_a**2 * mean_nj * mean_nj_inverse
    variance_group_component = 1/m * sigma_b** 2

    return variance_indy_component + variance_group_component

def get_variance_wococ_H(group_sizes, sigma_I, sigma_G):

    n = sum(group_sizes)
    m = len(group_sizes)

    sum_inverse_nj = sum([1/n_j for n_j in group_sizes])

    H = sum_inverse_nj/m**2 - 1/n

    variance_indy_component =  H * sigma_I**2  +  1/n * sigma_I**2
    variance_group_component = (1/m) * sigma_G** 2

    return variance_indy_component + variance_group_component


def get_variance(group_sizes, group_weights, sigma_i, sigma_j):

    N = sum(group_sizes)
    M = len(group_sizes)

    variance_indy_componnent = np.sum(group_weights**2/group_sizes) * sigma_i**2
    variance_group_component = np.sum(group_weights**2) * sigma_j**2

    return variance_indy_componnent + variance_group_component


def get_variance_2(group_sizes, group_weights, sigma_i, sigma_j):

    N = sum(group_sizes)
    M = len(group_sizes)

    betas = N / group_sizes

    variance_indy_componnent = 1/N * np.sum(betas * group_weights**2) * sigma_i**2
    variance_group_component = np.sum(group_weights**2) * sigma_j**2

    return variance_indy_componnent + variance_group_component

def get_variance_3(group_sizes, group_weights, sigma_i, sigma_j):

    N = sum(group_sizes)
    M = len(group_sizes)

    betas = N / group_sizes

    sigma_w = np.std(group_weights)

    variance_indy_componnent = 1/N * np.sum(betas * group_weights**2) * sigma_i**2
    variance_group_component = sigma_j**2 * (M * sigma_w**2 + 1/M)

    return variance_indy_componnent + variance_group_component

def get_variance_4(group_sizes, group_weights, sigma_i, sigma_j):

    N = sum(group_sizes)
    M = len(group_sizes)

    betas = N / group_sizes

    sigma_w = np.std(group_weights)

    variance_indy_componnent = 1/N * sigma_i**2 * (M*np.cov(betas, group_weights**2, bias=True)[0,1] + M*np.mean(betas)*sigma_w**2 + np.mean(betas) * 1/M)
    variance_group_component = sigma_j**2 * (M * sigma_w**2 + 1/M)
    return variance_indy_componnent + variance_group_component

def get_variance_5(group_sizes, group_weights, sigma_i, sigma_j):

    N = sum(group_sizes)
    M = len(group_sizes)

    betas = 1 / group_sizes

    sigma_w = np.std(group_weights)




    cov_beta_w_squared = np.cov(betas, group_weights**2, bias=True)[0,1]

    return (sigma_i**2 * np.mean(betas) + sigma_j**2) * (1/M + M * sigma_w**2) + sigma_i**2 * M * cov_beta_w_squared

def get_variance_6(group_sizes, group_weights, sigma_i, sigma_j):

    N = sum(group_sizes)
    M = len(group_sizes)

    betas = 1 / group_sizes

    alphas = group_weights * M
    sigma_alpha = np.std(alphas)
    sigma_w = np.std(group_weights)

    cov_beta_w_squared = np.cov(betas, group_weights**2, bias=True)[0,1]

    cov_beta_alpha_squared = np.cov(betas, alphas**2, bias=True)[0,1]


    return 1/M*((sigma_i**2 * np.mean(betas) + sigma_j**2) * (1 + sigma_alpha**2) + sigma_i**2 * cov_beta_alpha_squared)


def test_general_equation():

    group_sizes = np.array([1,2,1,3,1])
    sigma_i = 2
    sigma_j = 1

    group_weights_woc = group_sizes/sum(group_sizes)

    print("Variance with WOC weights:")
    print(get_variance(group_sizes, group_weights_woc, sigma_i, sigma_j))
    print(get_variance_woc(group_sizes, sigma_i, sigma_j))

    group_weights_wococ = np.ones(len(group_sizes))/len(group_sizes)

    print("Variance with WOCOC weights:")
    print(get_variance(group_sizes, group_weights_wococ, sigma_i, sigma_j))
    print(get_variance_wococ(group_sizes, sigma_i, sigma_j))

    group_weights_square_root = np.sqrt(group_sizes)/np.sum(np.sqrt(group_sizes))

    print("Variance with square root weights:")
    print(get_variance(group_sizes, group_weights_square_root, sigma_i, sigma_j))

    print(get_variance_2(group_sizes, group_weights_square_root, sigma_i, sigma_j))
    print(get_variance_3(group_sizes, group_weights_square_root, sigma_i, sigma_j))
    print(get_variance_4(group_sizes, group_weights_square_root, sigma_i, sigma_j))

    print(get_variance_5(group_sizes, group_weights_square_root, sigma_i, sigma_j))

    print(get_variance_6(group_sizes, group_weights_square_root, sigma_i, sigma_j))

    group_sizes = np.array([1,1,2,1])
    group_weights = np.array([0.25, 0.25, 0.25, 0.25])

    print("Variance with equal weights:")    
    group_weights_normed = group_weights/np.sum(group_weights)
    print(get_variance_5(group_sizes, group_weights_normed, sigma_i, sigma_j))

    group_weights = np.array([0.2, 0.2, 0.4, 0.2])
    print("Variance with unequal weights:")

    group_weights_normed = group_weights/np.sum(group_weights)
    print(get_variance_5(group_sizes, group_weights_normed, sigma_i, sigma_j))





sim_lots(1000)