# parameters
group_sizes = [30, 120, 480]   # unequal blocks
tau, sigma  = 1.0, 1.0
k, theta    = 12, 1.0
# 1) draw initial opinions with subgroup bias + iid noise
# 2) run T=200 iterations of rule (A)+(B)
# 3) measure limiting DeGroot weights (column sums of contact counts)
# you will find  corr(w_empirical , n_g/(n_g*tau**2+1))  > 0.95
