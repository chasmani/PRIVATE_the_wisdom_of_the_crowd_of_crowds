
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


def plot_coc_vs_flat(M, N, sigma_indy, sigma_group):
    
    Ds = np.linspace(1/M, 1, 100)
    H_invs = np.linspace(1/N, 1, 100)
    rho = sigma_indy**2/sigma_group**2
    A = N/M

    DD, HH = np.meshgrid(Ds, H_invs)

    # Calculate function values at each grid point
    ratio = get_var_ratio_coc_over_flat_compressed(N, M, DD, HH, rho)

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    
    divnorm=colors.TwoSlopeNorm(vmin=np.min(ratio), vcenter=1, vmax=np.max(ratio))
    im = plt.pcolormesh(Ds, H_invs, ratio, cmap="coolwarm", norm=divnorm)
    cb = plt.colorbar(im)
    cb.set_label(r'$\mathrm{Var}_{CC} / \mathrm{Var}_{C}$')

    H_equal = 1/A + (M * Ds - 1) / rho

    print(Ds, H_equal)

    plt.plot(Ds, H_equal, linewidth=2, label='Equality')

    plt.xlabel(r'Simpson Diversity Index $D = \frac{1}{N^2} \sum_j n_j^2$')
    plt.ylabel(r'Inverse Harmonic Group Size, $H^{-1} = \frac{1}{M} \sum_j \frac{1}{n_j}$')

    plt.xlim(1/M, 1)
    plt.ylim(1/N, 1)

    plt.show()

def plot_subplots():

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))

    rhos = [0.25, 1, 4]
    Ns = [20, 100, 400]
    M = 5

    resolution = 100
    ratios = np.zeros((len(rhos), len(Ns), resolution, resolution))

    for i, rho in enumerate(rhos):
        for j, N in enumerate(Ns):
            ax = axs[i, j]
            Ds = np.linspace(1/M, 1, 100)
            H_invs = np.linspace(1/N, 1, 100)

            DD, HH = np.meshgrid(Ds, H_invs)

            ratio = get_var_ratio_coc_over_flat_compressed(N, M, DD, HH, rho)

            ratios[i, j] = ratio

            """
            divnorm = colors.Normalize()

            im = ax.pcolormesh(Ds, H_invs, ratio, cmap="coolwarm", norm=divnorm)
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(r'$\mathrm{Var}_{CC} / \mathrm{Var}_{C}$') 

            H_equal = 1/M + (M * Ds - 1) / rho

            ax.plot(Ds, H_equal, linewidth=2, label='Equality')

            ax.set_xlim(1/M, 1)
            ax.set_ylim(1/N, 1)
            ax.set_title(f'ρ={rho}, N={N}')
            """

    vmin = np.min(ratios)
    vmax = np.max(ratios)

    for i, rho in enumerate(rhos):
        for j, N in enumerate(Ns):
            ax = axs[i, j]
            Ds = np.linspace(1/M, 1, 100)
            H_invs = np.linspace(1/N, 1, 100)
            ratio = ratios[i, j]    

            divnorm = colors.SymLogNorm(linthresh=1e-2, vmin=vmin, vmax=vmax, base=10)

            im = ax.pcolormesh(Ds, H_invs, ratio, cmap="coolwarm", norm=divnorm)

            H_equal = M/N + (M * Ds - 1) / rho
            ax.plot(Ds, H_equal, linewidth=2, label='Equality')

            ax.set_title(f'ρ={rho}, N={N}')
            ax.set_xlim(1/8, 1)
            ax.set_ylim(1/N, 1)

            if i == 2:
                ax.set_xlabel(r'Simpson Diversity $D = \frac{1}{N^2} \sum_j n_j^2$')
            if j == 0:
                ax.set_ylabel(r'Inverse Harmonic $H^{-1} = \frac{1}{M} \sum_j \frac{1}{n_j}$')

    # Generate some group samples
    for i, rho in enumerate(rhos):
        for j, N in enumerate(Ns):
            ax = axs[i, j]
            for n_groups in range(100):
                # Generate group sizes
                group_sizes = generate_group_sizes_uniform_composition_sampler(N, M)
                
                var_ratio = get_variance_ratio_coc_over_flat(group_sizes, sigma_indy=np.sqrt(rho), sigma_group=1)
                Ds = np.sum(np.array(group_sizes)**2) / N**2
                H_inv = np.mean(1/np.array(group_sizes))
                if var_ratio > 1:
                    color = 'red'
                else:
                    color = 'blue'
                ax.plot(Ds, H_inv, color=color, alpha=0.1, markeredgecolor=color, marker='o', markersize=5)

    im = axs[0, 0].collections[0]

    # Shrink subplots to make room on right
    fig.subplots_adjust(right=0.88)

    # Add a new axis for colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  
    cbar = fig.colorbar(im, cax=cbar_ax)
    tick_locs = [0.25, 0.5, 1, 2, 4]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([str(t) for t in tick_locs])
    cbar.set_label(r'var(crowd of crowds) / var(flat mean)')

    plt.savefig("images/coc_vs_flat.png", dpi=600, bbox_inches='tight')
    
    plt.show()


#plot_coc_vs_flat(M=5, N=400, sigma_indy=2, sigma_group=1)
plot_subplots()
