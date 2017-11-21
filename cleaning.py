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

def clean_and_divide(vertical):
	
	print "Cleaning "+str(vertical)+"."
	
	name_sales = "../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_data.csv"
	name_returns = "../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_data.csv"
	
	returns_order_item_id = []
	sales_id_date_dic = {}
	order_ids = []
	order_dates = []
	split_dates = []
	
	chunksize = 1000
	
	for df in pd.read_csv(name_sales, sep="\t", chunksize=chunksize):
		for index,rows in df.iterrows():
			sales_id_date_dic[str(rows['order_id'])] = str(rows['order_item_date'])
			order_ids.append(str(rows['order_id']))
			order_dates.append(str(rows['order_item_date']))
	
	for df in pd.read_csv(name_returns, sep="\t" , chunksize=chunksize , usecols=range(0,27)):
		for index,rows in df.iterrows():
			returns_order_item_id.append(str(rows['order_item_id']))
	
	order_dates.sort()
	
	returns_order_item_id = list(set(returns_order_item_id) & set(order_ids))
	
	split_dates.append(str(order_dates[int(len(order_dates)*0.80)]))
	split_dates.append(str(order_dates[int(len(order_dates)*0.90)]))
	
	if os.path.exists("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_training.csv"):
		os.remove("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_training.csv")
	if os.path.exists("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_validation.csv"):
		os.remove("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_validation.csv")
	if os.path.exists("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_testing.csv"):
		os.remove("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_testing.csv")
	
	if os.path.exists("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_training.csv"):
		os.remove("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_training.csv")
	if os.path.exists("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_validation.csv"):
		os.remove("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_validation.csv")
	if os.path.exists("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_testing.csv"):
		os.remove("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_testing.csv")
	
	sales_training = open("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_training.csv","a")
	sales_validation = open("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_validation.csv","a")
	sales_testing = open("../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_testing.csv","a")
	returns_training = open("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_training.csv","a")
	returns_validation = open("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_validation.csv","a")
	returns_testing = open("../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_testing.csv","a")
	
	print "Split values claculated."
	
	flag1 = 0
	flag2 = 0
	flag3 = 0
	
	for df in pd.read_csv(name_sales, sep="\t", chunksize=1):
		for index,rows in df.iterrows():
			if (str(rows['order_item_date']) <= split_dates[0]) and (str(rows['order_id']) not in returns_order_item_id):
				if flag1 == 0:
					df.to_csv(sales_training,sep="\t",index=False,header=True,na_rep="NULL")
					flag1 = 1
				else:
					df.to_csv(sales_training,sep="\t",header=False,index=False,na_rep="NULL")
			elif (str(rows['order_item_date']) > split_dates[0]) and (str(rows['order_item_date']) <= split_dates[1]):
				if flag2 == 0:
					df.to_csv(sales_validation,sep="\t",index=False,header=True,na_rep="NULL")
					flag2 = 1
				else:
					df.to_csv(sales_validation,sep="\t",header=False,index=False,na_rep="NULL")
			elif str(rows['order_item_date']) > split_dates[1]:
				if flag3 == 0:
					df.to_csv(sales_testing,sep="\t",index=False,header=True,na_rep="NULL")
					flag3 = 1
				else:
					df.to_csv(sales_testing,sep="\t",header=False,index=False,na_rep="NULL")
	print "Sales Spliting Done."
	
	flag1 = 0
	flag2 = 0
	flag3 = 0
	
	for df in pd.read_csv(name_returns, sep="\t" , chunksize=1 , usecols=range(0,27)):
		for index,rows in df.iterrows():
			if (str(rows['order_item_id']) in returns_order_item_id) and (sales_id_date_dic[str(rows['order_item_id'])] <= split_dates[0]):
				if flag1 == 0:
					df.to_csv(returns_training,sep="\t",index=False,header=True,na_rep="NULL")
					flag1 = 1
				else:
					df.to_csv(returns_training,sep="\t",header=False,index=False,na_rep="NULL")
			elif (str(rows['order_item_id']) in returns_order_item_id) and (sales_id_date_dic[str(rows['order_item_id'])] > split_dates[0]) and (sales_id_date_dic[str(rows['order_item_id'])] <= split_dates[1]):
				if flag2 == 0:
					df.to_csv(returns_validation,sep="\t",index=False,header=True,na_rep="NULL")
					flag2 = 1
				else:
					df.to_csv(returns_validation,sep="\t",header=False,index=False,na_rep="NULL")
			elif str(rows['order_item_id']) in returns_order_item_id:
				if flag3 == 0:
					df.to_csv(returns_testing,sep="\t",index=False,header=True,na_rep="NULL")
					flag3 = 1
				else:
					df.to_csv(returns_testing,sep="\t",header=False,index=False,na_rep="NULL")
	print "Returns Spliting Done."
	
	sales_training.close()
	sales_validation.close()
	sales_testing.close()
	returns_training.close()
	returns_validation.close()
	returns_testing.close()

def generate_graphs(vertical):
	
	for both in ["sales","returns"]:
		print "Generating Graphs of "+str(vertical)+" "+str(both)+"."
		
		data = "../data/"+str(vertical)+"/"+str(both)+"/data/"
		pickles = "../data/"+str(vertical)+"/"+str(both)+"/pickles/"
		name = data+str(vertical)+"_"+str(both)+"_training"
		
		size_column_name = 'size'
		if (vertical == "womenbellies") or (vertical == "mencasualshoes"):
			size_column_name = 'uk_india_size'
		
		till_col = None
		if both == "returns":
			till_col = range(0,27)
		
		if os.path.exists(name+"_edgelist.txt"):
			os.remove(name+"_edgelist.txt")
		
		file2 = open(name+"_edgelist.txt","w")
		
		G = nx.Graph()
		G_w = nx.Graph()
		bs = set()
		users = set()
		sizes = set()
		brands = set()
		
		count = 0
		
		chunksize = 1000
		for df in pd.read_csv(name+".csv", sep="\t",chunksize=chunksize,usecols=till_col):
			for index,rows in df.iterrows():
				if ((str(rows['brand']).lower()) == "null") or ((str(rows[size_column_name]).lower()) == "null"):
					continue
				
				count = count + 1
				bs.add((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())
				users.add(str(rows['account_id']))
				sizes.add(str(rows[size_column_name]).lower())
				brands.add(str(rows['brand']).lower())
				if not (G.has_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())):
					G.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())
				
				if not (G_w.has_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())):
					G_w.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower(),weight=1)
				else:
					G_w[rows['account_id']][(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()]['weight'] += 1
		
		if os.path.exists(name+"_stats.txt"):
			os.remove(name+"_stats.txt")
		
		stats = open(name+"_stats.txt","w")
		stats.write("Unique Users- "+str(len(users))+"\n")
		stats.write("Unique Brands- "+str(len(brands))+"\n")
		stats.write("Unique Sizes- "+str(len(sizes))+"\n")
		stats.write("Unique BSs- "+str(len(bs))+"\n")
		stats.close()
		
		name = pickles+str(vertical)+"_"+str(both)+"_training"
		
		bs = list(bs)
		bs.sort()
		
		if os.path.exists(name+"_bi.pickle"):
			os.remove(name+"_bi.pickle")
		if os.path.exists(name+"_bs.pickle"):
			os.remove(name+"_bs.pickle")
		if os.path.exists(name+"_bi_weighted.pickle"):
			os.remove(name+"_bi_weighted.pickle")
		
		
		nx.write_gpickle(G,name+"_bi.pickle")
		with open(name+"_bs.pickle",'w') as f:
			pickle.dump(bs,f)
		nx.write_gpickle(G_w,name+"_bi_weighted.pickle")
		
		print "Rows in training CSV: "+str(count)
		
		G2=nx.Graph()
		
		count = 0
		for pair in itertools.combinations(bs,2):
			temp = len(set(G.neighbors(pair[0])).intersection(G.neighbors(pair[1])))
			if temp > 1:
				G2.add_edge(pair[0],pair[1],weight=temp)
				count = count + 1
				file2.write(str(bs.index(pair[0]))+" "+str(bs.index(pair[1]))+" "+str(temp)+"\n")
		
		file2.close()
		
		if os.path.exists(name+"_pro.pickle"):
			os.remove(name+"_pro.pickle")
		nx.write_gpickle(G2,name+"_pro.pickle")
		
		print "Rows in training edgelist: "+str(count)
		print "Done with "+str(vertical)+" "+str(both)+"."

def main():
	verticals = ["mencasualshirt"]
	for vertical in verticals:
		clean_and_divide(vertical)
		generate_graphs(vertical)

if __name__ == '__main__':
	main()
