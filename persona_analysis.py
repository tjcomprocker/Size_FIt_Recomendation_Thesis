import pandas as pd
import numpy as np
import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("svg")

import matplotlib.pyplot as plt
plt.style.use("seaborn")


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
	
	for both in ["sales"]:
		print "Generating Persona Analysis of "+str(vertical)+" "+str(both)+"."
		
		G=nx.read_gpickle("../data/"+str(vertical)+"/"+str(both)+"/pickles/"+str(vertical)+"_"+str(both)+"_training_bi.pickle")
		
		vals = []
		sizes = set()
		
		for v in G.nodes():
			if str(v).find(";") == -1:
				temp = set()
				for n in G.neighbors_iter(v):
					temp.add((str(n).split(";"))[1])
				vals.append(len(list(temp)))
			else:
				sizes.add((str(v).split(";"))[1])
		
		sizes = list(sizes)
		for s in sizes:
			print s
		
		plt.hist(vals,bins=range(0,max(vals)))
		plt.yscale('log', nonposy='clip')
		plt.ylabel("No. of Users",fontsize=17)
		plt.xlabel("Unique Sizes Bought",fontsize=17)
		plt.xticks(np.arange(1,max(vals),1.0))
		plt.tick_params(axis='both', which='major', labelsize=17)
		plt.tight_layout()
		plt.savefig("../data/"+str(vertical)+"/"+str(both)+"/plots/persona_analysis.png")
		plt.clf()

def persona_binning_1(vertical):
	
	for both in ["sales"]:
		print "Generating Persona Analysis of "+str(vertical)+" "+str(both)+"."
		
		
		G=nx.read_gpickle("../data/"+str(vertical)+"/"+str(both)+"/pickles/"+str(vertical)+"_"+str(both)+"_training_bi.pickle")
		
		vals = []
		
		bins = [["nan"],["xxs","xs","s","2","4","6","8","10","12","14","16"],["m","22","23","24","25","26","27","28","29"],["free","l","xl","30","31","32","46/32","33","34","44/34","42/34","36","38"],["xxl","3xl","4xl","5xl","6xl","7xl","40","42","44","46","48","50","52"]]
		
		if str(vertical) == "mencasualshirt":
			bins = [["nan"],["xs","xxs","3xs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years","8"],["xl","l","el","xxl","free","40","42","44","46","48"],["m","s","30","31","32","34","36","37","38","39"],["3xl","4xl","5xl","6xl","7xl","50","52","54"]]
		elif str(vertical) == "menpolotshirt":
			bins = [["nan"],["xs","xxs","9 - 10 years","2","4","8","14"],["m","s","22","24","28","30","32","34","36","38","39","104","116","128"],["xl","l","el","xxl","free","40","42","44","46","48","140","152","164"],["3xl","4xl","5xl","6xl","7xl","8xl","50","52","54"]]
		
		
		total = 0
		count = 0
		
		for v in G.nodes():
			if str(v).find(";") == -1:
				temp = set()
				for n in G.neighbors_iter(v):
					for i in range(0,len(bins)):
						if (str(n).split(";"))[1] in bins[i]:
							temp.add(i)
				vals.append(len(list(temp)))
				if len(list(temp)) > 1:
					count += 1
				total += 1
		
		print "For binning-1 percentage of users which bought from more than one bins  is - "+str((float(count)/float(total))*100)
		
		plt.hist(vals,bins=range(1,max(vals)))
		plt.ylabel("No. of Users",fontsize=17)
		plt.xlabel("User bought from different size bins",fontsize=17)
		plt.xticks(np.arange(1,max(vals),1.0))
		plt.tick_params(axis='both', which='major', labelsize=17)
		plt.tight_layout()
		plt.savefig("../data/"+str(vertical)+"/"+str(both)+"/plots/persona_analysis_bins_1.png")
		plt.clf()

def persona_binning_2(vertical):
	
	for both in ["sales"]:
		print "Generating Persona Analysis of "+str(vertical)+" "+str(both)+"."
		
		G=nx.read_gpickle("../data/"+str(vertical)+"/"+str(both)+"/pickles/"+str(vertical)+"_"+str(both)+"_training_bi.pickle")
		
		vals = []
		
		bins = [["nan"],["xxs","2","4","6","8"],["xs","10","12","14","16"],["s","22","23","24","25"],["m","26","27","28","29"],["l","30","31","32","46/32","33"],["free","xl","34","44/34","42/34","36","38"],["xxl","3xl","40","42","44"],["4xl","5xl","6xl","7xl","46","48","50","52"]]
		
		if str(vertical) == "mencasualshirt":
			bins = [["nan"],["xs","xxs","3xs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years","8"],["l","el","40","42"],["xl","xxl","free","44","46","48"],["s","30","31","32","34"],["m","36","37","38","39"],["3xl","4xl","5xl","6xl","7xl","50","52","54"]]
		elif str(vertical) == "menpolotshirt":
			bins = [["nan"],["xs","xxs","9 - 10 years","2","4","8","14"],["m","s","22","24","28","30","32","34","36","38","39","104","116","128"],["xl","l","el","xxl","free","40","42","44","46","48","140","152","164"],["3xl","4xl","5xl","6xl","7xl","8xl","50","52","54"]]
		
		
		total = 0
		count = 0
		
		for v in G.nodes():
			if str(v).find(";") == -1:
				temp = set()
				for n in G.neighbors_iter(v):
					for i in range(0,len(bins)):
						if (str(n).split(";"))[1] in bins[i]:
							temp.add(i)
				#vals.append(len(list(temp)))
				if len(list(temp)) > 1:
					count += 1
				total += 1
		
		print "For binning-2 percentage of users which bought from more than one bins  is - "+str((float(count)/float(total))*100)
		
		"""plt.hist(vals,bins=range(1,max(vals)))
		plt.title("Persona Analysis of "+str(vertical)+" "+str(both)+" with bins_2.")
		plt.ylabel("No. of Users")
		plt.xlabel("Unique Bins Bought From")
		plt.savefig("../data/"+str(vertical)+"/"+str(both)+"/plots/persona_analysis_bins_2.png")
		plt.clf()"""

def bin_wise(vertical):
	vals = [[],[]]
	
	bins = [["nan"],["xxs","xs","s","2","4","6","8","10","12","14","16"],["m","22","23","24","25","26","27","28","29"],["free","l","xl","30","31","32","46/32","33","34","44/34","42/34","36","38"],["xxl","3xl","4xl","5xl","6xl","7xl","40","42","44","46","48","50","52"]]
	if str(vertical) == "mencasualshirt":
		bins = [["nan"],["xs","xxs","3xs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years","8"],["xl","l","el","xxl","free","40","42","44","46","48"],["m","s","30","31","32","34","36","37","38","39"],["3xl","4xl","5xl","6xl","7xl","50","52","54"]]
	elif str(vertical) == "menpolotshirt":
		bins = [["nan"],["xs","xxs","9 - 10 years","2","4","8","14"],["m","s","22","24","28","30","32","34","36","38","39","104","116","128"],["xl","l","el","xxl","free","40","42","44","46","48","140","152","164"],["3xl","4xl","5xl","6xl","7xl","8xl","50","52","54"]]
	
	for both in ["sales","returns"]:
		print "Generating Bin-wise Analysis of "+str(vertical)+" "+str(both)+"."
		
		
		G=nx.read_gpickle("../data/"+str(vertical)+"/"+str(both)+"/pickles/"+str(vertical)+"_"+str(both)+"_training_bi.pickle")
		
		for v in G.nodes():
			if str(v).find(";") != -1:
				for i in range(1,len(bins)):
					if (str(v).split(";"))[1] in bins[i]:
						vals[["sales","returns"].index(both)].append(i-1)
	
	#print vals
	
	plt.hist(vals, 5, histtype='bar')
	plt.ylabel("No. of Users that bought/returned from a bin",fontsize=17)
	plt.xticks([0,1,2,3],["bin-1","bin-2","bin-3","bin-4"])
	plt.tick_params(axis='both', which='major', labelsize=17)
	plt.tight_layout()
	blue_patch = mpatches.Patch(color='blue', label='Sales')
	green_patch = mpatches.Patch(color='green', label='Returns')
	plt.legend(handles=[blue_patch,green_patch],prop={'size': 17})
	plt.savefig("../data/"+str(vertical)+"/sales/plots/binswise_"+str(vertical)+".png")
	plt.clf()

def top_brands(vertical):
	for both in ["sales","returns"]:
		print "Top brands for "+str(vertical)+" "+str(both)+"."
		
		G=nx.read_gpickle("../data/"+str(vertical)+"/"+str(both)+"/pickles/"+str(vertical)+"_"+str(both)+"_training_bi.pickle")
		
		brand_dic = {}
		
		for v in G.nodes():
			if str(v).find(";") != -1:
				if (str(v).split(";"))[0] in brand_dic:
					brand_dic[(str(v).split(";"))[0]] += len(G.neighbors(v))
				else:
					brand_dic[(str(v).split(";"))[0]] = 0
		
		brand_list = []
		for key, value in brand_dic.iteritems():
			brand_list.append((value,key))
		
		brand_list.sort(reverse=True)
		
		for b in brand_list[:5]:
			print b[1]

def main():
	verticals = ["womenjean","mencasualshirt","menpolotshirt"]
	for vertical in verticals:
		#persona(vertical)
		persona_binning_1(vertical)
		#persona_binning_2(vertical)
		#bin_wise(vertical)
		#top_brands(vertical)

if __name__ == '__main__':
	main()
