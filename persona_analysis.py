import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import csv
import itertools
import operator
import sys
import pickle
from random import randint
import subprocess
import os
import seaborn as sns

def persona(vertical):
	
	for both in ["sales","returns"]:
		print "Generating Perosna Analysis of "+str(vertical)+" "+str(both)+"."
		
		if os.path.exists("../data/"+str(vertical)+"/"+str(both)+"/plots/"):
			os.system("rm -rf ../data/"+str(vertical)+"/"+str(both)+"/plots/")
		os.system("mkdir ../data/"+str(vertical)+"/"+str(both)+"/plots/")
		
		G=nx.read_gpickle("../data/"+str(vertical)+"/"+str(both)+"/pickles/"+str(vertical)+"_"+str(both)+"_training_bi.pickle")
		
		vals = []
		
		for v in G.nodes():
			if str(v).find(";") == -1:
				temp = set()
				for n in G.neighbors(v):
					temp.add((str(n).split(";"))[1])
				vals.append(len(list(temp)))
		
		plt.hist(vals,bins=range(1,max(vals)))
		plt.yscale('log', nonposy='clip')
		plt.title("Perona Analysis of "+str(vertical)+" "+str(both)+".")
		plt.ylabel("No. of Users")
		plt.xlabel("Unique Sizes Bought")
		plt.savefig("../data/"+str(vertical)+"/"+str(both)+"/plots/persona_analysis.png")
		plt.clf()

def main():
	verticals = ["womenjean","mencasualshirt"]
	for vertical in verticals:
		persona(vertical)

if __name__ == '__main__':
	main()
