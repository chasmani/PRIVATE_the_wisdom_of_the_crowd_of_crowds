
import numpy as np
import networkx as nx


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

def build_homophily_network_with_closure(opinions):
	"""
	Builds a homophily network with triadic closure.
	Nodes connect based on opinion similarity and shared neighbors.
	"""
	G = nx.DiGraph()
	G.add_nodes_from(range(len(opinions)))

	degree_target = 2

	for i in range(len(opinions)):
		# Base homophily probabilities
		ps_homophily = np.exp(-0.1 * abs(opinions[i] - np.array(opinions)))
		ps_homophily = ps_homophily / sum(ps_homophily)

		# Combine with triadic closure
		ps_combined = np.zeros(len(opinions))
		for j in range(len(opinions)):
			# Base homophily score
			ps_combined[j] = ps_homophily[j]

			# Add triadic closure component if i and j share neighbors
			if j in G:
				common_neighbors = set(G.neighbors(i)).intersection(set(G.neighbors(j)))
				if len(common_neighbors) > 0:
					ps_combined[j] += 0.5 * len(common_neighbors)

		# Normalize combined probabilities
		ps_combined = ps_combined / sum(ps_combined)

		# Select target nodes for new edges
		js = np.random.choice(range(len(opinions)), degree_target, p=ps_combined, replace=False)
		for j in js:
			G.add_edge(i, j)

	# Print adjacency matrix
	print(nx.adjacency_matrix(G).todense())

	return G


def build_homophily_network(opinions):

	G = nx.DiGraph()

	G.add_nodes_from(range(len(opinions)))

	degree_out_target = 5

	for i in range(len(opinions)):
		ps = np.exp(-abs(opinions[i] - np.array(opinions)))
		ps = ps/sum(ps)
		print(ps)
		# Choose random nodes to connect to
		js = np.random.choice(range(len(opinions)), degree_out_target, p=ps, replace=False)
		for j in js:
			G.add_edge(i, j)

	return G

def triadic_closure(G):

	edges_to_add = []

	# Iterate over all nodes in the graph
	for node in G.nodes:
		# Get neighbors of the current node
		neighbors = set(G.neighbors(node))

		# Iterate over all pairs of neighbors (potential closure candidates)
		for neighbor1 in neighbors:
			for neighbor2 in neighbors:
				if neighbor1 != neighbor2 and not G.has_edge(neighbor1, neighbor2):
					# Form a new edge with the specified probability
					if np.random.rand() < 0.5:
						edges_to_add.append((neighbor1, neighbor2))

	# Add all new edges to the graph
	G.add_edges_from(edges_to_add)

	print(edges_to_add)

	return G

def count_open_triads_each_node(G):

	open_triads_nodes = []

	for node in G.nodes:
		open_triads = 0
		neighbors = list(G.neighbors(node))

		for i in range(len(neighbors)):
			for j in range(i + 1, len(neighbors)):
				# If these neighbors are not connected, we found an open triad
				if not G.has_edge(neighbors[i], neighbors[j]):
					open_triads += 1

		open_triads_nodes.append(open_triads)

	return open_triads_nodes


def sim_one_network_digraph():

	group_sizes = [8, 4]
	sigma_indy = 2
	sigma_group = 7
	truth = 0
	seed = 2

	opinions = generate_opinions(group_sizes, sigma_indy, sigma_group, truth, seed=None)
	woc = np.mean(opinions)
	
	G = build_homophily_network(opinions)


	# Get leading eigenvector of adjacency matrix
	A = nx.adjacency_matrix(G).todense()
	print(A)

	# Normalise A
	A = A / np.sum(A, axis=0)

	W = get_eigenweights(A.T)

	print(W)


def build_homophilly_complete_weights(opinions):

	n = len(opinions)

	A = np.zeros((n,n))

	for i in range(n):
		ps = np.exp(-abs(opinions[i] - np.array(opinions)))
		ps = ps/sum(ps)
		for j in range(n):
			A[i,j] = ps[j]
	
	return A

def sim_one_network_weighted():

	group_sizes = [20, 40]
	sigma_indy = 2
	sigma_group = 0
	truth = 0
	seed = 0
	np.random.seed(seed)

	opinions = generate_opinions(group_sizes, sigma_indy, sigma_group, truth, seed=None)
	woc = np.mean(opinions)
	
	A = build_homophilly_complete_weights(opinions)

	# Normalise A
	A = A / np.sum(A, axis=0)

	# Print outweight

	W = get_eigenweights(A)


	w_1 = 1/(group_sizes[0]*sigma_group**2 + sigma_indy**2)
	w_2 = 1/(group_sizes[1]*sigma_group**2 + sigma_indy**2)
	print(w_1, w_2)
	print("Optimal weights 1 vs 2", w_1/w_2)

	# Get group 1 average weight
	group_1_weight = np.sum(W[:group_sizes[0]])/group_sizes[0]
	group_2_weight = np.sum(W[group_sizes[0]:])/group_sizes[1]
	print("Netowrk weights group 1 vs 2", group_1_weight/group_2_weight)

	# Get square root weights
	sqrt_w_1 = 1/np.sqrt(group_sizes[0])
	sqrt_w_2 = 1/np.sqrt(group_sizes[1])

	print("SQuare root weights 1 vs 2", sqrt_w_1/sqrt_w_2)

	# WOC weights
	print("Wisdom of crowd erights is ", 1)

	# WOCOC wieghts 
	print("Wisdom of crowd of crowds weights is ", 1/ (group_sizes[0]/group_sizes[1]))


def get_eigenweights(A):

	# Check if all cols sum to 1
	if not np.allclose(np.sum(A, axis=0), 1):
		print("Rows do not sum to 1")
		return None

	# Find the leading eigenvector
	eigenvalues, eigenvectors = np.linalg.eig(A)

	leading_eigenvector_ind = np.argmax(eigenvalues)
	leading_eigenvector = eigenvectors[:,leading_eigenvector_ind]

	# Normalise the leading eigenvector
	normalised_eigenvector = leading_eigenvector/sum(leading_eigenvector)

	return np.real(normalised_eigenvector)

"""
A = np.matrix([
	[0.25, 0.25, 0.25, 0.25],
	[0.25, 0.25, 0.25, 0.25],
	[0.2, 0, 0.8, 0],
	[0, 0.2, 0, 0.8]])


A = np.matrix([
	[0.3, 0.3, 0.3],
	[0.3, 0.3, 0.3],
	[0.25, 0.25, 0.5]])


A = np.matrix([
	[1, 0, 1, 0, 1],
	[0, 1, 0, 1, 0],
	[0, 1, 1, 0, 0],
	[1, 0, 1, 1, 1],
	[1, 1, 0, 1, 1]])

A = np.matrix([
	[0.333, 0, 0.333, 0, 0.333],
	[0, 0.5, 0, 0.5, 0],
	[0, 0.5, 0.5, 0, 0],
	[0.25, 0, 0.25, 0.25, 0.25],
	[0.25, 0.25, 0, 0.25, 0.25]])

A = np.matrix([
	[1, 1, 1, 0, 1],
	[1, 1, 1, 1, 0],
	[1, 1, 1, 0, 0],
	[1, 0, 1, 1, 1],
	[0, 1, 0, 1, 1]])

A = np.matrix([
	[0.25, 0.25, 0.25, 0, 0.25],
	[0.25, 0.25, 0.25, 0.25, 0],
	[0.25, 0.25, 0.25, 0.25, 0],
	[0.333, 0, 0, 0.333, 0.333],
	[0, 0.333, 0, 0.333, 0.333]])

	"""



if __name__=="__main__":
	sim_one_network_weighted()
