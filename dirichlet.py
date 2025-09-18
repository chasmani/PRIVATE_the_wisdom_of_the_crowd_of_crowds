
import numpy as np
from scipy.stats import dirichlet

M = 20
kappa = 0.9
N = 100

# Symmetri Dirichlet
alpha = [kappa] * M
ps = dirichlet.rvs(alpha, size=1)[0]
print(ps)

S = np.sum(ps**2)
H = 1/(N*M) * np.mean(1/ps)
print(S, H)

E_S = (kappa + 1)/(M * kappa + 1)
E_H = (M*kappa - 1)/(kappa - 1) * 1/(M*N)

print(E_S, E_H)

r_star = (S - 1/M)/(H-1/N)
print(r_star)

E_r = (E_S - 1/M)/(E_H - 1/N)
print(E_r)