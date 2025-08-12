

sigma_group = 1
sigma_indy = 1

n_groups = [10,10,10,10,10]
n = sum(n_groups)
print(n)

variance_group_component = sigma_group**2 * sum([(n_i/n)**2 for n_i in n_groups])
variance_indy_component = 1/n * sigma_indy**2 

print(variance_group_component, variance_indy_component)
print(variance_group_component + variance_indy_component)

n_groups[0] += 1
n = sum(n_groups)

variance_group_component = sigma_group**2 * sum([(n_i/n)**2 for n_i in n_groups])
variance_indy_component = 1/n * sigma_indy**2 
print(variance_group_component, variance_indy_component)
print(variance_group_component + variance_indy_component)


def variance_wisdom_crowd(n_groups, sigma_group, sigma_indy):
    n = sum(n_groups)
    variance_group_component = sigma_group**2 * sum([(n_i/n)**2 for n_i in n_groups])
    variance_indy_component = 1/n * sigma_indy**2 
    return variance_group_component + variance_indy_component

def variance_wisdom_crowd_of_crowds(n_groups, sigma_group, sigma_indy):
    n = sum(n_groups)
    m = len(n_groups)

    variance_group_component = 1/m * sigma_group**2
    variance_indy_component = 1/m**2 * sigma_indy** 2 * sum([1/n_i for n_i in n_groups])
    return variance_group_component + variance_indy_component

n_groups = [20,10,10,10,10]

print(variance_wisdom_crowd(n_groups, 1, 1))
print(variance_wisdom_crowd_of_crowds(n_groups, 1, 1))

n_groups = [21,10,10,10,10]

print(variance_wisdom_crowd(n_groups, 1, 1))
print(variance_wisdom_crowd_of_crowds(n_groups, 1, 1))

# FINDINGS
# 1. Bias is neutral in both cases. 
# 2. Variance is different. 
# 3. Efficiency of esitmators depends on the subgroup structure and relative variances of the group and individual level.
# 4. Effect fo ading new individuals:
# a. Can actually reduce efficiency by adding new individauls to biggets group in wisdom of crowd. 
# b. The wisdom of the crowd of crowds estimator always improves by adding new individuals, but will improve more when adding individuals to small (or new) group.
