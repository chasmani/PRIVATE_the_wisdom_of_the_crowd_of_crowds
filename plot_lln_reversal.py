

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

def append_to_csv(row, filename="data/lln_reversal_simulation_results.csv"):

    """
    Append a row to a CSV file.
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)    

def sim_pref_attachment(group_sizes, alpha=1.5):

    gamma = 1

    ps = (group_sizes + gamma)**alpha
    ps = ps / np.sum(ps)

    j = np.random.choice(len(group_sizes), p=ps)

    group_sizes[j] += 1

    return group_sizes

def initial_population(initial_group_sizes, sigma_G, sigma_I):

    group_biases = np.random.normal(0, sigma_G, len(initial_group_sizes))

    population = [list() for _ in initial_group_sizes]

    for i, group_size in enumerate(initial_group_sizes):
        group_bias = group_biases[i]
        for _ in range(group_size):
            indy_bias = np.random.normal(0, sigma_I)
            new_opinion = group_bias + indy_bias
            population[i].append(new_opinion)

    return population, group_biases

def run_basic_sim():

    sigma_G = 1
    sigma_I = 2

    initial_group_sizes = [10,10,10,10,10]

    population, group_biases = initial_population(initial_group_sizes, sigma_G, sigma_I)
    
    group_sizes = [len(group) for group in population]
    print("WOC variance:", get_variance_woc(group_sizes, sigma_I, sigma_G))
    print("WOCOC variance:", get_variance_wococ(group_sizes, sigma_I, sigma_G))

    for T in range(100):
        population = sim_pref_attachment(population, group_biases, sigma_I, sigma_G)
        group_sizes = [len(group) for group in population]
        
    print("WOC variance:", get_variance_woc(group_sizes, sigma_I, sigma_G))
    print("WOCOC variance:", get_variance_wococ(group_sizes, sigma_I, sigma_G))

    for T in range(1000):
        population = sim_pref_attachment(population, group_biases, sigma_I, sigma_G)
        group_sizes = [len(group) for group in population]

    print(group_sizes)
    print("WOC variance:", get_variance_woc(group_sizes, sigma_I, sigma_G))
    print("WOCOC variance:", get_variance_wococ(group_sizes, sigma_I, sigma_G))


def get_woc_estimate(population):
    """
    Calculate the Wisdom of Crowds estimate based on the population.
    This is a simple average of all opinions in the population.
    """
    all_opinions = [opinion for group in population for opinion in group]
    return np.mean(all_opinions)

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


def get_variance_min(group_sizes, sigma_I, sigma_G):
    """
    Calculate the variance of the minimum estimate.
    This is a simplified version assuming independence.
    """
    n = sum(group_sizes)
    m = len(group_sizes)

    n_js = np.array(group_sizes)

    w_js = 1 / (sigma_G**2 + sigma_I**2 / n_js)
    w_js = w_js / np.sum(w_js)

    indy_component = sigma_I**2 * np.sum(w_js**2/n_js)
    group_component = sigma_G**2 * np.sum(w_js**2)

    return indy_component + group_component

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

def run_sims(sigma_I=1, data_filename="data/lln_reversal_simulation_results.csv"):

    alphas = [1.5,1,0.5]
    N = 20
    M = 5
    sigma_G = 1
    seed = np.random.randint(0, 10000)

    initial_group_sizes = generate_group_sizes_uniform_composition_sampler(N, M)
    print("Initial group sizes:", initial_group_sizes)

    N_snapshopts = [20, 50, 100,200,500,1000, 2000, 5000, 10000, 20000, 50000, 100000]

    for alpha in alphas:

        group_sizes = initial_group_sizes.copy()

        for N in range(N_snapshopts[0], N_snapshopts[-1] + 1):
            
            if N in N_snapshopts:
                var_woc = get_variance_woc(group_sizes, sigma_I, sigma_G)
                var_wococ = get_variance_wococ(group_sizes, sigma_I, sigma_G)
                var_w_optimal = get_variance_min(group_sizes, sigma_I, sigma_G)
                var_wococ_sqrt = get_variance_wococ_sqrt(group_sizes, sigma_I, sigma_G)

                csv_row = [alpha, seed, group_sizes, sigma_G, sigma_I, N, M, var_woc, var_wococ, var_w_optimal]
                append_to_csv(csv_row, filename=data_filename)
                
            group_sizes = sim_pref_attachment(group_sizes, alpha=alpha)

def run_lots_of_sims(sigma_I=1, n_sims=100, data_filename="data/lln_reversal_simulation_results.csv"):

    for _ in range(n_sims):
        run_sims(sigma_I=sigma_I, data_filename=data_filename)
        print(_)


def plot_sims(data_filename="data/lln_reversal_simulation_results_b.csv"):

    CSV_PATH = data_filename
    ALPHAS = [1.5, 1, 0.5]     # panels left→right
    EST_COLS = {
        "WOC": ("mean_woc", "sd_woc", "WoC"),
        "WOCOC": ("mean_wococ", "sd_wococ", "WoCoC"),
        "W Optimal": ("mean_wopt", "sd_wopt", "Optimal"),
    }

    # --- load & clean ---
    # If your file ALREADY has headers, keep header=0. If not, uncomment the 'names=' line instead.
    df = pd.read_csv(
        CSV_PATH,
        names=['Alpha','Seed','Group Sizes','Sigma G','Sigma I','N','M','WOC','WOCOC','W Optimal']
    )

    # basic cleaning & typing
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Alpha','N','WOC','WOCOC','W Optimal'])
    df['Alpha'] = df['Alpha'].astype(float)
    df['N'] = df['N'].astype(int)

    # --- aggregate across seeds per (Alpha, N) ---
    agg = (
        df.groupby(['Alpha','N'])
        .agg(
            mean_woc=('WOC','mean'),   sd_woc=('WOC','std'),
            mean_wococ=('WOCOC','mean'), sd_wococ=('WOCOC','std'),
            mean_wopt=('W Optimal','mean'), sd_wopt=('W Optimal','std'),
        )
        .reset_index()
    )

# --- plotting (3 panels) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, a in zip(axes, ALPHAS):
        print(ax,a)
        
        dat = agg[agg['Alpha'] == a].sort_values('N')

        jitter = -1
        for col, (mean_col, sd_col, label) in EST_COLS.items():
            if col == "W Optimal":
                jitter = 1.02
                color = 'black'
                marker = "o"
            elif col == "WOC":
                jitter = 1  
                color = 'blue'
                marker = "s"
            elif col == "WOCOC":
                jitter = 0.98       
                color = 'orange'
                marker = "D"
            ax.plot(dat['N'], dat[mean_col], color=color, marker=marker, label=label, alpha=0.7)
            ax.errorbar(dat['N'] * jitter, dat[mean_col], yerr=dat[sd_col], 
                        fmt='none', capsize=4, color=color, alpha=0.7)


        ax.set_xscale("log")
        ax.set_xlabel('N (total population size)')
        ax.set_title(f'α = {a}')
        ax.grid(True, which='both', linewidth=0.3, alpha=0.6)
        # Set y-axis bottom to zero
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=10)

    axes[0].set_ylabel('Variance of Estimator')

    axes[0].set_title("Preferential Attachment")
    axes[1].set_title("Proportional Growth")
    axes[2].set_title("Diversity Seeking")

    axes[-1].legend(loc='upper right', frameon=False)
    fig.tight_layout()

    # Annotate with a,b,c
    for i, ax in enumerate(axes):
        ax.annotate(chr(97 + i), xy=(0.02, 1.02), xycoords='axes fraction', fontsize=24, fontweight='bold')


    plt.savefig("images/lln_reversal_sims.png", dpi=600, bbox_inches='tight')
    plt.show()



def run_and_plot():

    data_filename="data/lln_reversal_simulation_results_i.csv"
    #run_lots_of_sims(sigma_I=3.33333333, n_sims=100, data_filename=data_filename)
    plot_sims(data_filename=data_filename)


if __name__ == "__main__":
    run_and_plot()