import csv
import networkx as nx
from collections import defaultdict
import numpy as np
#from enthought.mayavi import mlab
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools as it
import community
from gephistreamer import graph
#from gephistreamer import streame 

def getConnections(content): 
	f = {}
	for i in xrange(len(content)): 
		if content[i][0] not in f.keys():
			f[content[i][0]] = []
			f[content[i][0]].append((content[i][1]))
		else: 
			f[content[i][0]].append(content[i][1])
	return f

#####################################################################
#####################################################################
#####################################################################
#https://blog.dominodatalab.com/social-network-analysis-with-networkx/
def partitions(nodes, n): 
	nodes_iter = iter(nodes)
	while True: 
		partition = tuple(itertools.islice(nodes_iter, n))
		if not partition:
			return
		yield partition

def between_parallel(G, processes=None):
	p = Pool(processes=processes)
	part_generator = 4*len(p._pool)
	node_partitions = list(partitions(G.nodes(), int(len(G)/part_generator)))
	num_partitions = len(node_partitions)

	bet_map = p.map(btwn_pool, 
		zip([G]*num_partitions, 
			[True]*num_partitions, 
			[None]*num_partitions))

	bt_c = bet_map[0]
	for bt in bet_map[1:]: 
		for n in bt:
			bt_c[n]+= bt[n]
	return bt_c
################################################
################################################
################################################
################################################

def create_hc(G):
  #Creates hierarchical cluster of graph G from distance matrix
  path_length=nx.all_pairs_shortest_path_length(G)
  distances=numpy.zeros((len(G),len(G)))
  for u,p in path_length.items():
      for v,d in p.items():
          distances[u][v]=d
  # Create hierarchical cluster
  Y=distance.squareform(distances)
  Z=hierarchy.complete(Y)  # Creates HC using farthest point linkage
  # This partition selection is arbitrary, for illustrive purposes
  membership=list(hierarchy.fcluster(Z,t=1.15))
  # Create collection of lists for blockmodel
  partition=defaultdict(list)
  for n,p in zip(list(range(len(G))),membership):
      partition[p].append(n)
  return list(partition.values())

def main(FILE): 
	rows = []
	with open(FIL, 'r') as f: 
		reader=csv.reader(f)
		for row in reader: 
			rows.append(row)
	d= getConnections(rows)
	g = nx.Graph()
	for i in xrange(len(d.keys())):
		for j in xrange(len(d[d.keys()[i]])): 
			g.add_edge(str(d.keys()[i]), d[d.keys()[i]][j], weight=0.5)
	print "whee"

	#get the parameters we want 
	deg = nx.degree_centrality(g)
	d = nx.degree(g)
	#clique = nx.graph_clique_number(g) 
	cliques=[clique for clique in nx.find_cliques(g) if len(clique)>50]
	#tried with 2, was bad...
	#print clique
	#k = nx.k_components(g) 
	#clust = nx.clustering(g)

#	con = nx.all_pairs_node_connectivity(g)
#	local_con = nx.local_node_connectivity(g)
#	g_con = nx.node_connectivity(g)
#	print "Cons", con, local_con, g_con

	info = nx.info(g)
	print "Info", info
#	c = nx.average_clustering(g)
#	print "Average Clustering: ", c

	close = nx.closeness_centrality(g, u=None, distance=None, normalized=True)
	btwn = nx.betweenness_centrality(g, k=None, normalized=True, endpoints=False, seed=None)	
	nx.set_node_attributes(g, 'betweenness', btwn)
	nx.set_node_attributes(g, 'closeness', close)
	nx.set_node_attributes(g, 'degree', deg)
	close_avg = 0
	for c in xrange(len(close.values())): 
		close_avg+=close.values()[c]
	float(close_avg)/len(close.values())
	print "Closeness Average", close_avg

	bet = 0
	for b in xrange(len(btwn.values())): 
		bet += btwn.values()[b]
	float(bet)/len(btwn.values())
	print "Betweenness Average", bet

	#draws the basic visualization
	nx.draw(g, pos=nx.spring_layout(g, k=.05, iterations=20), 
		cmap = plt.get_cmap('Paired'), 
		#node_size=80, 
		node_size=[v for v in d.values()],
		node_color=[v for v in close.values()], 
		figsize=(19, 7))
	plt.show() 
	return 

FIL = "C:/Users/skyet/OneDrive/CMU/70-311/skye.toor.facebookdata.csv"
main(FIL)
