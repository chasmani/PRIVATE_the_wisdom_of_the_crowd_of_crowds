import matplotlib.pyplot as plt
import numpy as np


def optimal_weights_prop_to(n_j, sigma_I, sigma_G):

    return 1/(sigma_I**2/n_j + sigma_G**2)

def plot_optimal():

    n_js = np.linspace(1, 100, 100)
    sigma_i, sigma_j = 1,1

    optimal_weights = [optimal_weights_prop_to(n_j, sigma_i, sigma_j) for n_j in n_js]

    
    #plt.plot(n_js, optimal_weights)



    # Transofmr x-axis to square root
    plt.xscale('log')

    plt.xlabel('n_j')

    plt.ylabel('Optimal weight')

    plt.title('Optimal weight as a function of n_j')

    plt.show()

plot_optimal()


