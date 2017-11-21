from igraph import *
import sys

vertical = sys.argv[1]

for both in ["sales","returns"]:
	with open("../../data/"+str(vertical)+"/"+str(both)+"/data/"+str(vertical)+"_"+str(both)+"_training_edgelist.txt") as f:
		content = f.readlines()
	
	elist = []
	
	edge_w = []
	for lines in content:
		edge_w.append(map(int,lines[:-1].split("\t"))[2])
		elist.append((map(int,lines[:-1].split("\t"))[0],map(int,lines[:-1].split("\t"))[1]))
	
	g = Graph.Read_Ncol("../../data/"+str(vertical)+"/"+str(both)+"/data/"+str(vertical)+"_"+str(both)+"_training_edgelist.txt",weights=True,directed=False)
	
	names = g.vs["name"]
	
	communities = g.community_fastgreedy(weights = edge_w)
	clusters = communities.as_clustering()
	
	fl = open("../../data/"+str(vertical)+"/results/method_3/fastgreedy/"+str(both)+"/clusters.txt",'w')
	
	for x in clusters:
		for y in x:
			fl.write(str(names[y])+"\t")
		fl.write("\n")
	
	fl.close()
	
	communities = g.community_walktrap(weights = edge_w)
	clusters = communities.as_clustering()
	
	fl = open("../../data/"+str(vertical)+"/results/method_3/walktrap/"+str(both)+"/clusters.txt",'w')
	
	for x in clusters:
		for y in x:
			fl.write(str(names[y])+"\t")
		fl.write("\n")
	
	fl.close()
	
	communities = g.community_infomap(edge_weights = edge_w)
	
	fl = open("../../data/"+str(vertical)+"/results/method_3/infomap/"+str(both)+"/clusters.txt",'w')
	
	for x in communities:
		for y in x:
			fl.write(str(names[y])+"\t")
		fl.write("\n")
	
	fl.close()
