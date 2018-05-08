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
		
		attrs = ["color","pattern"]
		
		if str(vertical) == "mencasualshirt":
			attrs = ["color","pattern","occasion","sleeve"]
		
		if str(vertical) == "menpolotshirt":
			attrs = ["color","type","pattern","occasion","sleeve"]
		
		file2 = open(name+"_edgelist.txt","w")
		
		G = nx.Graph()
		G_w = nx.Graph()
		G_a = nx.MultiGraph()
		bs = set()
		users = set()
		sizes = set()
		brands = set()
		
		count = 0
		
		chunksize = 1000
		for df in pd.read_csv(name+".csv", sep="\t",chunksize=chunksize,usecols=till_col):
			for index,rows in df.iterrows():
				if ((str(rows['brand']).lower()) == "nan") or ((str(rows[size_column_name]).lower()) == "nan"):
					continue
				
				count = count + 1
				bs.add((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())
				users.add(str(rows['account_id']))
				sizes.add(str(rows[size_column_name]).lower())
				brands.add(str(rows['brand']).lower())
				if not (G.has_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())):
					G.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())
				
				flag = 0
				
				if str(vertical) == "womenjean":
					if not (G_a.has_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())):
						G_a.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower(),color=str(rows['color']).lower(),pattern=str(rows['pattern']).lower())
					else:
						check_edge = {"color":str(rows['color']).lower(),"pattern":str(rows['pattern']).lower()}
						for x in G_a[rows['account_id']][(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()].values():
							if cmp(x,check_edge) == 0:
								flag = 1
								break
						if flag == 0:
							G_a.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower(),color=str(rows['color']).lower(),pattern=str(rows['pattern']).lower())
				
				elif str(vertical) == "mencasualshirt":
					if not (G_a.has_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())):
						G_a.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower(),color=str(rows['color']).lower(),pattern=str(rows['pattern']).lower(),occasion=str(rows['occasion']).lower(),sleeve=str(rows['sleeve']).lower())
					else:
						check_edge = {"color":str(rows['color']).lower(),"pattern":str(rows['pattern']).lower(),"occasion":str(rows['occasion']).lower(),"sleeve":str(rows['sleeve']).lower()}
						for x in G_a[rows['account_id']][(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()].values():
							if cmp(x,check_edge) == 0:
								flag = 1
								break
						if flag == 0:
							G_a.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower(),color=str(rows['color']).lower(),pattern=str(rows['pattern']).lower(),occasion=str(rows['occasion']).lower(),sleeve=str(rows['sleeve']).lower())
				
				elif str(vertical) == "menpolotshirt":
					if not (G_a.has_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())):
						G_a.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower(),color=str(rows['color']).lower(),type=str(rows['type']).lower(),pattern=str(rows['pattern']).lower(),occasion=str(rows['occasion']).lower(),sleeve=str(rows['sleeve']).lower())
					else:
						check_edge = {"color":str(rows['color']).lower(),"type":str(rows['type']).lower(),"pattern":str(rows['pattern']).lower(),"occasion":str(rows['occasion']).lower(),"sleeve":str(rows['sleeve']).lower()}
						for x in G_a[rows['account_id']][(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()].values():
							if cmp(x,check_edge) == 0:
								flag = 1
								break
						if flag == 0:
							G_a.add_edge(rows['account_id'],(str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower(),color=str(rows['color']).lower(),type=str(rows['type']).lower(),pattern=str(rows['pattern']).lower(),occasion=str(rows['occasion']).lower(),sleeve=str(rows['sleeve']).lower())
				
				
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
		
		print "No of nodes in attributes graph "+str(G_a.number_of_nodes())
		print "No of edges in attributes graph "+str(G_a.number_of_edges())
		
		name = pickles+str(vertical)+"_"+str(both)+"_training"
		
		bs = list(bs)
		bs.sort()
		
		if os.path.exists(name+"_bi.pickle"):
			os.remove(name+"_bi.pickle")
		if os.path.exists(name+"_bs.pickle"):
			os.remove(name+"_bs.pickle")
		if os.path.exists(name+"_bi_weighted.pickle"):
			os.remove(name+"_bi_weighted.pickle")
		if os.path.exists(name+"_bi_atr.pickle"):
			os.remove(name+"_bi_atr.pickle")
		
		
		nx.write_gpickle(G,name+"_bi.pickle")
		with open(name+"_bs.pickle",'w') as f:
			pickle.dump(bs,f)
		nx.write_gpickle(G_w,name+"_bi_weighted.pickle")
		nx.write_gpickle(G_a,name+"_bi_atr.pickle")
		print "Rows in training CSV: "+str(count)
		
		G2=nx.Graph()
		
		count = 0
		for pair in itertools.combinations(bs,2):
			temp = len(set(G.neighbors(pair[0])).intersection(G.neighbors(pair[1])))
			if temp >= 1:
				G2.add_edge(pair[0],pair[1],weight=temp)
				count = count + 1
				file2.write(str(bs.index(pair[0]))+"\t"+str(bs.index(pair[1]))+"\t"+str(temp)+"\n")
		
		file2.close()
		
		if os.path.exists(name+"_pro.pickle"):
			os.remove(name+"_pro.pickle")
		nx.write_gpickle(G2,name+"_pro.pickle")
		
		print "Rows in training edgelist: "+str(count)
		print "Done with "+str(vertical)+" "+str(both)+"."


def generate_graphs_with_bins(vertical):
	
	for both in ["sales","returns"]:
		print "Generating Graphs of "+str(vertical)+" "+str(both)+"."
		
		data = "../data/"+str(vertical)+"/"+str(both)+"/data/"
		pickles = "../data/"+str(vertical)+"/"+str(both)+"/pickles/"
		name = data+str(vertical)+"_"+str(both)+"_training"
		
		bins = {"nan":[],"xxs":["xs","2","4","6","8","10","12"],"2":["xxs","xs","4","6","8"],"4":["xxs","xs","2","6","8","10"],"6":["xxs","xs","2","4","8","10"],"8":["xs","4","6","10","12"],"xs":["xxs","2","4","6","8","10","12","14","16"],"10":["xs","4","6","8","12","14","16"],"12":["s","8","10","14","16"],"14":["s","8","10","12","16"],"16":["s","10","12","14","22","23"],"s":["12","14","16","22","23","24","25"],"22":["s","16","23","24","25"],"23":["s","m","16","22","24","25","26"],"24":["s","m","22","23","25","26"],"25":["s","m","23","24","26","27"],"m":["s","23","24","25","26","27","28","29","30"],"26":["m","24","25","27","28","29","30"],"27":["m","l","24","25","26","28","29","30","31"],"28":["m","l","25","26","27","29","30","31"],"29":["m","l","26","27","28","30","31","32","46/32"],"l":["27","28","29","30","31","32","46/32"],"30":["l","28","29","31","32","46/32","33"],"31":["l","28","29","20","32","46/32","33"],"32":["l","29","30","31","46/32","33"],"46/32":["l","free","29","30","31","32","33","34","44/34","42/34"],"33":["l","free","30","31","32","46/32","34","44/34","42/34","36"],"free":["l","xl","30","31","32","33","46/32","34","44/34","42/34","36","38"],"xl":["l","free","30","31","32","33","46/32","34","44/34","42/34","36","38"],"34":["l","free","31","32","33","46/32","34","44/34","42/34","36"],"44/34":["l","free","xl","31","32","33","46/32","34","42/34","36","38"],"42/34":["l","free","xl","31","32","33","46/32","34","44/34","36","38"],"36":["l","free","xl","32","33","46/32","34","44/34","38","40"],"38":["xl","xxl","34","44/34","36","40","42"],"xxl":["xl","38","40","42"],"3xl":["xl","xxl","40","42","44","46","48"],"40":["xl","xxl","36","38","42","44"],"42":["xl","xxl","3xl","38","40","44","46"],"44":["xxl","3xl","40","42","46","48"],"4xl":["3xl","5xl","44","46","48","50","52"],"5xl":["4xl","6xl","44","46","48","50","52"],"6xl":["5xl","7xl","44","46","48","50","52"],"7xl":["6xl","44","46","48","50","52"],"46":["3xl","4xl","5xl","42","44","48","50"],"48":["4xl","5xl","6xl","44","46","50","52"],"50":["5xl","6xl","7xl","46","48","52"],"52":["5xl","6xl","7xl","46","48","50"]}
		
		if str(vertical) == "mencasualshirt":
			bins = {"nan":[],"s":["xs","xxs","3xs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years","8"],"xs":["s","xxs","3xs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years","8"],"xxs":["s","xs","3xs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years","8"],"3xs":["s","xs","xxs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years","8"],"2 - 3 years":["s","xs","xxs","3xs","8"],"3 - 4 years":["s","xs","xxs","3xs","8"],"4 - 5 years":["s","xs","xxs","3xs","8"],"5 - 6 years":["s","xs","xxs","3xs","8"],"6 - 7 years":["s","xs","xxs","3xs","8"],"7 - 8 years":["s","xs","xxs","3xs","8"],"8 - 9 years":["s","xs","xxs","3xs","8"],"9 - 10 years":["s","xs","xxs","3xs","8"],"11 - 12 years":["s","xs","xxs","3xs","8"],"8":["s","xs","xxs","3xs","2 - 3 years","3 - 4 years","4 - 5 years","5 - 6 years","6 - 7 years","7 - 8 years","8 - 9 years","9 - 10 years","11 - 12 years"],"m":["28","30","31","32","34","36","37","38","39"],"28":["m","30","31","32","34"],"30":["m","28","31","32","34"],"31":["m","28","30","32","34"],"32":["m","28","30","31","34","36"],"34":["m","30","31","32","36","37","38"],"36":["m","32","34","37","38","39","l","el","40"],"37":["m","34","36","38","39","l","el","40"],"38":["m","34","36","37","39","l","el","40","42"],"39":["m","34","36","37","38","l","el","40","42"],"l":["36","37","38","39","el","40","42","44","xl","free"],"el":["36","37","39","38","l","40","42","44","xl","free"],"40":["36","37","38","39","l","el","42","44","xl","free"],"42":["36","37","38","39","l","el","40","44","xl","free"],"xl":["l","el","40","42","44","46","48","free"],"xxl":["xl","free","44","46","48","50"],"2xl":["xl","free","44","46","48","50"],"free":["l","el","40","42","44","46","xl","xxl","2xl"],"44":["40","42","46","48","l","el","xl","free","xxl","2xl"],"46":["50","42","44","48","xl","free","xxl"],"48":["50","52","44","46","xl","3xl","xxl"],"3xl":["50","52","48","4xl","xxl","54"],"4xl":["50","52","3xl","5xl","54"],"5xl":["50","52","4xl","6xl","54"],"6xl":["50","52","5xl","7xl","54"],"7xl":["50","52","6xl","54"],"50":["3xl","4xl","5xl","6xl","7xl","52","54","46","48"],"52":["3xl","4xl","5xl","6xl","7xl","50","54","48"],"54":["3xl","4xl","5xl","6xl","7xl","52","50","48"]}
		
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
				if ((str(rows['brand']).lower()) == "nan") or ((str(rows[size_column_name]).lower()) == "nan"):
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
			if temp>=1 and (((pair[1].split(";"))[1] in bins[(pair[0].split(";"))[1]]) or ((pair[0].split(";"))[1] in bins[(pair[1].split(";"))[1]])):
				G2.add_edge(pair[0],pair[1],weight=temp)
				count = count + 1
				file2.write(str(bs.index(pair[0]))+"\t"+str(bs.index(pair[1]))+"\t"+str(temp)+"\n")
		
		file2.close()
		
		if os.path.exists(name+"_pro.pickle"):
			os.remove(name+"_pro.pickle")
		nx.write_gpickle(G2,name+"_pro.pickle")
		
		print "Rows in training edgelist: "+str(count)
		print "Done with "+str(vertical)+" "+str(both)+"."

def attributes(vertical):
	for both in ["sales","returns"]:
		print "Attribute analysis "+str(vertical)+" "+str(both)+"."
		
		data = "../data/"+str(vertical)+"/"+str(both)+"/data/"
		name = data+str(vertical)+"_"+str(both)+"_training"
		
		size_column_name = 'size'
		if (vertical == "womenbellies") or (vertical == "mencasualshoes"):
			size_column_name = 'uk_india_size'
		
		till_col = None
		if both == "returns":
			till_col = range(0,27)
		
		count_total = 0
		count = {"color":0 , "weight":0 , "length":0 , "breadth":0 , "height":0 ,"type":0 , "variant":0 , "design":0 , "pattern":0 , "occasion":0 , "outer_material":0 , "material":0 , "sleeve":0 }
		chunksize = 1000
		for df in pd.read_csv(name+".csv", sep="\t",chunksize=chunksize,usecols=till_col):
			for index,rows in df.iterrows():
				if ((str(rows['brand']).lower()) == "nan") or ((str(rows[size_column_name]).lower()) == "nan"):
					continue
				
				count_total += 1
				for attr in ["color","weight","length","breadth","height","type","variant","design","pattern","occasion","outer_material","material","sleeve"]:
					if ((str(rows[attr]).lower()) == "null") or ((str(rows[attr]).lower()) == "na") or ((str(rows[attr]).lower()) == "") or ((str(rows[attr]).lower()) == "nan"):
						count[attr] = count[attr] + 1
		print count_total
		print count
		
		for attr in ["color","weight","length","breadth","height","type","variant","design","pattern","occasion","outer_material","material","sleeve"]:
			print "In "+attr+" "+str((float(count[attr])/count_total)*100)+"% of entries are empty."
		
def main():
	verticals = ["menpolotshirt"]
	for vertical in verticals:
		#clean_and_divide(vertical)
		#generate_graphs(vertical)
		attributes(vertical)

if __name__ == '__main__':
	main()
