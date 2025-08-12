
import ast

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import csv


def append_to_csv(csv_list, output_filename):
    with open(output_filename, 'a', newline='') as fp:
        a = csv.writer(fp, delimiter=';')
        data = [csv_list]
        a.writerows(data)

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



def woc(opinions):
    return np.mean(opinions)

def wococ(opinions, group_sizes):

    m = len(group_sizes)
    n = sum(group_sizes)

    group_means = []

    for j in range(m):
        n_j = group_sizes[j]
        group_means.append(np.mean(opinions[:n_j]))
        opinions = opinions[n_j:]

    wococ = np.mean(group_means)

    return wococ

def wococ_sqrt(opinions, group_sizes):

    m = len(group_sizes)
    n = sum(group_sizes)

    group_means = []
    group_weights = []

    for j in range(m):
        n_j = group_sizes[j]
        group_means.append(np.mean(opinions[:n_j]))
        opinions = opinions[n_j:]
        group_weights.append(np.sqrt(n_j))

    wococ_sqrt = np.average(group_means, weights=group_weights)

    return wococ_sqrt

def wococ_optimal(opinions, group_sizes, sigma_I, sigma_G):

    m = len(group_sizes)
    n = sum(group_sizes)

    group_means = []
    group_weights = []

    for j in range(m):
        n_j = group_sizes[j]
        group_means.append(np.mean(opinions[:n_j]))
        opinions = opinions[n_j:]

        group_weight = 1/(sigma_G**2 + sigma_I**2/n_j)

        group_weights.append(group_weight)

    wococ_optimal = np.average(group_means, weights=group_weights)

    return wococ_optimal

def wococ_optimal_estimated(opinions, group_sizes):

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



def sim_data():

    group_sizes = [100, 100]
    sigma_indy = 10
    sigma_group = 1

    opinions = generate_opinions(group_sizes, sigma_indy, sigma_group, seed=6)

    print("WOC:", woc(opinions))
    print("WOCOC:", wococ(opinions, group_sizes))
    print("WOCOC SQRT:", wococ_sqrt(opinions, group_sizes))
    print("WOCOC Optimal:", wococ_optimal(opinions, group_sizes, sigma_indy, sigma_group))
    print("WOCOC Optimal Estimated Vars:", wococ_optimal_estimated_vars(opinions, group_sizes))

    return opinions


def empirical_variance(group_sizes, sigma_I, sigma_G, num_samples = 100):

    wocs = []
    wococs = []
    wococs_sqrt_vars = []
    wococs_optimal_vars = []
    wococs_optimal_estimated_vars = []

    for _ in range(1, num_samples):
        opinions = generate_opinions(group_sizes, sigma_I, sigma_G, seed=None)

        wocs.append(woc(opinions))
        wococs.append(wococ(opinions, group_sizes))
        wococs_sqrt_vars.append(wococ_sqrt(opinions, group_sizes))
        wococs_optimal_vars.append(wococ_optimal(opinions, group_sizes, sigma_I, sigma_G))
        wococs_optimal_estimated_vars.append(wococ_optimal_estimated(opinions, group_sizes))

    woc_var = np.var(wocs)
    wococ_var = np.var(wococs)
    wococ_sqrt_var = np.var(wococs_sqrt_vars)
    wococ_optimal_var = np.var(wococs_optimal_vars)
    wococ_optimal_estimated_var = np.var(wococs_optimal_estimated_vars)

    return woc_var, wococ_var, wococ_sqrt_var, wococ_optimal_var, wococ_optimal_estimated_var

def get_harmonic_mean(group_sizes):

    m = len(group_sizes)
    return 1/m * sum([1/x for x in group_sizes])


def sim_lots_of_data():

    xs = []

    woc_vars = []
    wococ_vars = []
    wococ_sqrt_vars = []
    wococ_optimal_vars = []
    wococ_optimal_estimated_vars = []

    max_x = 1

    for seed in range(100):
        # randomise group sizes
        np.random.seed(seed)
        print(seed)
        group_sizes = np.random.randint(1, 1000, 5)
        sigma_indy = 1
        sigma_group = 0.1

        harmonic_mean = get_harmonic_mean(group_sizes)

        x = sigma_group**2/sigma_indy**2 + harmonic_mean

        if x > max_x:
            print(group_sizes, sigma_indy, sigma_group)
            max_x = x

        xs.append(x)

        woc_var, wococ_var, wococ_sqrt_var, wococ_optimal_var, wococ_optimal_estimated_var = empirical_variance(group_sizes, sigma_indy=sigma_indy, sigma_group=sigma_group, seed=None)

        woc_vars.append(woc_var)
        wococ_vars.append(wococ_var)
        wococ_sqrt_vars.append(wococ_sqrt_var)
        wococ_optimal_vars.append(wococ_optimal_var)
        wococ_optimal_estimated_vars.append(wococ_optimal_estimated_var)


    #plt.scatter(xs, woc_vars, label="WOC", marker="x")
    plt.scatter(xs, wococ_vars, label="WOCOC", marker="o")
    plt.scatter(xs, wococ_sqrt_vars, label="WOCOC SQRT", marker="^")
    #plt.scatter(xs, wococ_optimal_vars, label="WOCOC Optimal", marker="s")
    #plt.scatter(xs, wococ_optimal_estimated_vars, label="WOCOC Optimal w Empirical Vars", marker="d")

    plt.xlabel(r"$\sigma_G^2 + \sigma_I^2 \frac{1}{M} \sum \frac{1}{n_j}$")
    plt.ylabel(r"var($\hat{θ}$)")
    plt.legend()


    plt.savefig("images/sims_and_variances_random_without_woc_{}.png".format(seed+1), dpi=300)

    plt.scatter(xs, woc_vars, label="WOC", marker="x")
    plt.legend()

    plt.savefig("images/sims_and_variances_random_{}.png".format(seed+1), dpi=300)


    plt.show()

def sim_and_save_one_run(keyword, group_sizes, sigma_I, sigma_G, num_var_samples, seed, output_save_file="data/sims_output_basic.csv"):

    np.random.seed(seed)

    harmonic_mean = get_harmonic_mean(group_sizes)

    mean_group_var = sigma_G**2 + sigma_I**2 * harmonic_mean

    woc_var, wococ_var, wococ_sqrt_var, wococ_optimal_var, wococ_optimal_estimated_var = empirical_variance(group_sizes, sigma_I=sigma_I, sigma_G=sigma_G, num_samples=num_var_samples)
    
    csv_row = [keyword, seed, sigma_I, sigma_G, group_sizes, mean_group_var, woc_var, wococ_var, wococ_sqrt_var, wococ_optimal_var, wococ_optimal_estimated_var]

    append_to_csv(csv_row, output_save_file)

    return csv_row


    
def sim_and_save_many():

    keyword = "random_10_v_small_groups"
    for seed in range(100):
        group_sizes = np.random.randint(1, 10, 5)
        sigma_I = 1
        for sigma_G in [0.15, 15]:

            print("Running", sigma_G, seed)

            sim_and_save_one_run(keyword, group_sizes, sigma_I, sigma_G, num_var_samples=100, seed=seed, output_save_file="data/sims_output_basic.csv")


    
def plot_from_file(keyword=None):


    df = pd.read_csv("data/sims_output_basic.csv", delimiter=";", names=["keyword", "seed", "sigma_I", "sigma_G", "group_sizes", "mean_group_var", "woc_var", "wococ_var", "wococ_sqrt_var", "wococ_optimal_var", "wococ_optimal_estimated_var"])

    if keyword:
        df = df[df["keyword"]==keyword]

    # Get unique rows
    df = df.drop_duplicates()

    df["harmonic_mean"] = df["mean_group_var"] - df["sigma_G"]**2

    # eval group sizes as numpy list
    print(df["group_sizes"])
    df["group_sizes"] = df['group_sizes'].str.strip('[]').str.split().apply(lambda x: np.array(x, dtype=int))
    
    
    # Get std of inverse group sizes
    # Get inverse group sizes
    df["inv_group_sizes"] = df["group_sizes"].apply(lambda x: 1/x)
    df["inv_group_sizes_std"] = df["inv_group_sizes"].apply(lambda x: np.std(x))



    # Normalise variances by dividing by wococ_optimal_var
    
    df["woc_var"] = df["woc_var"] / df["wococ_optimal_var"]
    df["wococ_var"] = df["wococ_var"] / df["wococ_optimal_var"]
    df["wococ_sqrt_var"] = df["wococ_sqrt_var"] / df["wococ_optimal_var"]
    df["wococ_optimal_estimated_var"] = df["wococ_optimal_estimated_var"] / df["wococ_optimal_var"]
    
    df["normed_mean_group_var"] = df["mean_group_var"] / df["inv_group_sizes_std"]

    x_axis = "normed_mean_group_var"

    # Plot hitograms with central tendency and variation in bins
    #sns.histplot(df, x=x_axis, y="woc_var", bins=100, kde=True, label="WOC", color="blue")
    #sns.histplot(df, x=x_axis, y="wococ_var", bins=100, kde=True, label="WOCOC", color="red")

    # Create log bins along x-axis 
    bins = np.logspace(np.log10(df[x_axis].min()), np.log10(df[x_axis].max()), 100)
    df["bin"] = np.digitize(df[x_axis], bins)

    # Get central tendency and variation in each bin
    central_tendency = df.groupby("bin").mean()
    variation = df.groupby("bin").std()

    plt.errorbar(central_tendency[x_axis], central_tendency["woc_var"], label="WOC", fmt="x")
    plt.errorbar(central_tendency[x_axis], central_tendency["wococ_var"], label="WOCOC", fmt="o")
    plt.errorbar(central_tendency[x_axis], central_tendency["wococ_sqrt_var"], label="WOCOC SQRT", fmt="^")
    plt.errorbar(central_tendency[x_axis], central_tendency["wococ_optimal_estimated_var"], label="WOCOC Optimal w Empirical Vars", fmt="d")


    #sns.regplot(x=x_axis, y="woc_var", data=df, label="WOC", marker="x", x_ci="sd", scatter_kws={"s": 10, "alpha":0.1})
    #sns.regplot(x=x_axis, y="wococ_var", data=df, label="WOCOC", marker="o", x_ci="sd", scatter_kws={"s": 10, "alpha":0.1})
    #sns.regplot(x=x_axis, y="wococ_sqrt_var", data=df, label="WOCOC SQRT", marker="^", x_ci="sd", scatter_kws={"s": 10, "alpha":0.1})
    #sns.regplot(x=x_axis, y="wococ_optimal_estimated_var", data=df, label="WOCOC Optimal w Empirical Vars", marker="d", x_ci="sd", scatter_kws={"s": 10, "alpha":0.1})


    #plt.scatter(df[x_axis], df["woc_var"], label="WOC", marker="x", s=5, alpha=0.2)
    #plt.scatter(df[x_axis], df["wococ_var"], label="WOCOC", marker="o", s=5, alpha=0.2)
    #plt.scatter(df[x_axis], df["wococ_sqrt_var"], label="WOCOC SQRT", marker="^", s=5, alpha=0.2)
    #plt.scatter(df[x_axis], df["wococ_optimal_estimated_var"], label="WOCOC Optimal w Empirical Vars", marker="d", s=5, alpha=0.2)

    # Plot scatterplots with line fits
    #plt.scatter(df[x_axis], df["woc_var"], label="WOC", marker="x")

    plt.xlabel(r"$\frac{1}{\sigma_{\beta}} * (\sigma_G^2 + \sigma_I^2 \frac{1}{M} \sum \frac{1}{n_j})$")

    plt.ylabel(r"Relative var($\hat{θ}$)")
    plt.legend()

    # log scale x axis
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("images/sims_bins_all_relative_var.png", dpi=300)


    plt.show()


if __name__=="__main__":
    #sim_and_save_many()
    plot_from_file()