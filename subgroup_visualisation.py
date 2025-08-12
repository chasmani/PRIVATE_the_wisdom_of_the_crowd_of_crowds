import numpy as np
import matplotlib.pyplot as plt


def plot_well_seperated():

    # Parameters
    np.random.seed(42)
    num_groups = 4
    points_per_group = [10, 5, 50, 15]  # unbalanced group sizes
    group_means = np.sort(np.random.uniform(0, 100, num_groups))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    noise_scale = 0.7  # tight grouping

    # Generate data
    x = np.concatenate([
        np.random.normal(loc=mean, scale=noise_scale, size=size)
        for mean, size in zip(group_means, points_per_group)
    ])
    y = np.random.normal(0, 0.05, sum(points_per_group))  # small vertical jitter

    # Plot
    plt.figure(figsize=(10, 3))
    start_idx = 0
    for idx, (mean, size, color) in enumerate(zip(group_means, points_per_group, colors)):
        end_idx = start_idx + size
        plt.scatter(x[start_idx:end_idx], y[start_idx:end_idx], alpha=0.7,
                    color=color, label=f'$u_{idx+1}$')
        plt.text(mean, 0.15, f'$u_{idx+1}$', ha='center', fontsize=22, color=color)
        start_idx = end_idx

    plt.yticks([])
    plt.xticks([])


    # Remove axes
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)


    plt.xlabel(r'$y_{ij}$', fontsize=22)

    plt.savefig("images/seperated_subgroups.png", dpi=600)

    plt.show()


def plot_not_well_seperated():

    # Parameters
    np.random.seed(42)
    num_groups = 4
    points_per_group = [10, 5, 50, 15]  # unbalanced group sizes
    group_means = np.sort(np.random.uniform(0, 100, num_groups))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    noise_scale = 20  # tight grouping

    # Generate data
    x = np.concatenate([
        np.random.normal(loc=mean, scale=noise_scale, size=size)
        for mean, size in zip(group_means, points_per_group)
    ])
    y = np.random.normal(0, 0.05, sum(points_per_group))  # small vertical jitter

    # Plot
    plt.figure(figsize=(10, 3))
    start_idx = 0
    for idx, (mean, size, color) in enumerate(zip(group_means, points_per_group, colors)):
        end_idx = start_idx + size
        plt.scatter(x[start_idx:end_idx], y[start_idx:end_idx], alpha=0.7,
                    color=color, label=f'$u_{idx+1}$')
        plt.text(mean, 0.15, f'$u_{idx+1}$', ha='center', fontsize=22, color=color)
        start_idx = end_idx

    plt.yticks([])
    plt.xticks([])


    # Remove axes
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)


    plt.xlabel(r'$y_{ij}$', fontsize=22)

    plt.savefig("images/unseperated_subgroups.png", dpi=600)

    plt.show()

plot_well_seperated()