import pickle
import networkx as nx
import pandas as pd
import numpy as np
import random
import sys
import subprocess
import os
import glob
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

def clustering(vertical,k):
	
	print "Clustering "+str(vertical)+" with Method-4."
	
	"""if os.path.exists("../../data/"+str(vertical)+"/results/method_4/"):
		os.system("rm -r ../../data/"+str(vertical)+"/results/method_4/")"""
	os.system("mkdir ../../data/"+str(vertical)+"/results/method_4/")
	
	os.system("mkdir ../../data/"+str(vertical)+"/results/method_4/sales/")
	
	os.system("mkdir ../../data/"+str(vertical)+"/results/method_4/returns/")
	
	os.system("python lib/src/main.py --input ../../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_training_edgelist.txt --output vecs_sales.txt --weighted")
	
	os.system("python lib/src/main.py --input ../../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_training_edgelist.txt --output vecs_returns.txt --weighted")
	
	sales_vecs = []
	sales_bs = []
	returns_vecs = []
	returns_bs = []
	
	with open("vecs_sales.txt","r") as f:
		content = f.readlines()
		for lines in content[1:]:
			temp = map(float,(lines.split(" ")))
			sales_vecs.append(temp[1:])
			sales_bs.append(int(temp[0]))
	
	with open("vecs_returns.txt","r") as f:
		content = f.readlines()
		for lines in content[1:]:
			temp = map(float,(lines.split(" ")))
			returns_vecs.append(temp[1:])
			returns_bs.append(int(temp[0]))
	
	kmeans_sales = KMeans(n_clusters=k).fit(sales_vecs)
	clus_sales = (kmeans_sales.labels_).tolist()
	kmeans_returns = KMeans(n_clusters=k).fit(returns_vecs)
	clus_returns = (kmeans_returns.labels_).tolist()
	
	sales_file = []
	returns_file = []
	
	for i in range(0,k):
		sales_file.append([])
	
	for i in range(0,k):
		returns_file.append([])
	
	for i in range(0,len(sales_bs)):
		sales_file[clus_sales[i]].append(sales_bs[i])
	
	for i in range(0,len(returns_bs)):
		returns_file[clus_returns[i]].append(returns_bs[i])
	
	fl = open("../../data/"+str(vertical)+"/results/method_4/sales/clusters.txt",'w')
	
	for x in sales_file:
		for y in x:
			fl.write(str(y)+"\t")
		fl.write("\n")
	
	fl.close()
	
	fl = open("../../data/"+str(vertical)+"/results/method_4/returns/clusters.txt",'w')
	
	for x in returns_file:
		for y in x:
			fl.write(str(y)+"\t")
		fl.write("\n")
	
	fl.close()
	
	os.system("rm vecs_returns.txt")
	os.system("rm vecs_sales.txt")


def evaluation(vertical):
	
	print "Training of "+str(vertical)+" begins."
	
	cluster_sales = []
	cluster_sales_flat = []
	
	cluster_returns = []
	cluster_returns_flat = []
	
	sales_bs = list(pickle.load(open("../../data/"+str(vertical)+"/sales/pickles/"+str(vertical)+"_sales_training_bs.pickle","r")))
	sales_bs.sort()
	returns_bs = list(pickle.load(open("../../data/"+str(vertical)+"/returns/pickles/"+str(vertical)+"_returns_training_bs.pickle","r")))
	returns_bs.sort()
	
	with open("../../data/"+str(vertical)+"/results/method_4/sales/clusters.txt","r") as f:
		content = f.readlines()
		for lines in content:
			cluster_sales.append(map(int,(lines.split("\t"))[:-1]))
	
	cluster_sales_flat = [sales_bs[item] for sublist in cluster_sales for item in sublist]
	
	with open("../../data/"+str(vertical)+"/results/method_4/returns/clusters.txt","r") as f:
		content = f.readlines()
		for lines in content:
			cluster_returns.append(map(int,(lines.split("\t"))[:-1]))
	
	cluster_returns_flat = [returns_bs[item] for sublist in cluster_returns for item in sublist]
	
	if str(vertical) == "womenbellies" or str(vertical) == "mencasualshoes":
		size_column_name = 'uk_india_size'
	else:
		size_column_name = 'size'
	
	sales_validation_name = "../../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_validation.csv"
	returns_validation_name = "../../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_validation.csv"
	
	sales_testing_name = "../../data/"+str(vertical)+"/sales/data/"+str(vertical)+"_sales_testing.csv"
	returns_testing_name = "../../data/"+str(vertical)+"/returns/data/"+str(vertical)+"_returns_testing.csv"
	
	G_sales_bi_weighted = nx.Graph()
	G_sales_bi_weighted = nx.read_gpickle("../../data/"+str(vertical)+"/sales/pickles/"+str(vertical)+"_sales_training_bi_weighted.pickle")
	
	G_returns_bi_weighted = nx.Graph()
	G_returns_bi_weighted = nx.read_gpickle("../../data/"+str(vertical)+"/returns/pickles/"+str(vertical)+"_returns_training_bi_weighted.pickle")
	
	predicted_sales = []
	predicted_returns = []
	predicted_combo = []
	for_combo = []
	for_combo2 = []
	ground_truth_sales_method = []
	ground_truth_returns_method = []
	ground_truth_combo_method = []
	
	ground_truth_returns = []
	
	not_clustered_sales = 0
	user_cold_start_sales = 0
	total_not_null_sales = 0
	
	chunksize = 1000
	for df in pd.read_csv(returns_validation_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			ground_truth_returns.append((rows['order_item_id'],rows['brand'],rows['account_id'],rows[size_column_name]))
	
	for df in pd.read_csv(sales_validation_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			
			total_not_null_sales += 1
			
			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in sales_bs and (str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower() in cluster_sales_flat :
				clus = [x for x in cluster_sales if sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in x][0]
			
			else:
				not_clustered_sales += 1
				continue
			
			if not G_sales_bi_weighted.has_node(str(rows['account_id'])):
				user_cold_start_sales += 1
				continue
			
			if G_returns_bi_weighted.has_node(str(rows['account_id'])):
				numerator = 0
				denominator = 0
				running_sum = 0
				total_sold = 0
				for items in clus:
					if G_returns_bi_weighted.has_edge(str(rows['account_id']),sales_bs[items]):
						numerator = G_returns_bi_weighted[str(rows['account_id'])][sales_bs[items]]['weight']
						denominator = numerator
					else:
						numerator = 0
					if G_sales_bi_weighted.has_edge(str(rows['account_id']),sales_bs[items]):
						denominator += G_sales_bi_weighted[str(rows['account_id'])][sales_bs[items]]['weight']
					elif denominator == 0:
						continue
					running_sum = float(running_sum) + ((float(numerator))/float(denominator))
					if numerator > 0:
						total_sold += 1
				if total_sold > 0:
					predicted_sales.append((float(running_sum))/float(total_sold))
					for_combo.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
				else:
					predicted_sales.append(0)
					for_combo.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			else:
				predicted_sales.append(0)
				for_combo.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			
			if (rows['order_id'],rows['brand'],rows['account_id'],rows[size_column_name]) in ground_truth_returns:
				ground_truth_sales_method.append(1)
			else:
				ground_truth_sales_method.append(0)
	
	print "Sales Predictions Done."
	
	not_clustered_returns = 0
	user_cold_start_returns = 0
	total_not_null_returns = 0
	
	for df in pd.read_csv(sales_validation_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			
			total_not_null_returns += 1
			
			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in returns_bs and (str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower() in cluster_returns_flat:
				clus = [x for x in cluster_returns if returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in x][0]
			else:
				not_clustered_returns += 1
				continue
			
			if not G_sales_bi_weighted.has_node(str(rows['account_id'])):
				user_cold_start_returns += 1
				continue
			
			if G_returns_bi_weighted.has_node(str(rows['account_id'])):
				numerator = 0
				denominator = 0
				running_sum = 0
				total_sold = 0
				for items in clus:
					if G_returns_bi_weighted.has_edge(str(rows['account_id']),returns_bs[items]):
						numerator = G_returns_bi_weighted[str(rows['account_id'])][returns_bs[items]]['weight']
						denominator = numerator
					else:
						numerator = 0
					
					if G_sales_bi_weighted.has_edge(str(rows['account_id']),returns_bs[items]):
						denominator += G_sales_bi_weighted[str(rows['account_id'])][returns_bs[items]]['weight']
					elif denominator == 0:
						continue
					running_sum = float(running_sum) + ((float(numerator))/float(denominator))
					if numerator > 0:
						total_sold += 1
				if total_sold > 0:
					predicted_returns.append((float(running_sum))/float(total_sold))
					for_combo2.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
				else:
					predicted_returns.append(0)
					for_combo2.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			else:
				predicted_returns.append(0)
				for_combo2.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			
			if (rows['order_id'],rows['brand'],rows['account_id'],rows[size_column_name]) in ground_truth_returns:
				ground_truth_returns_method.append(1)
			else:
				ground_truth_returns_method.append(0)
	
	for df in pd.read_csv(sales_validation_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			
			temp_tuple = (str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id']))
			if temp_tuple in for_combo and temp_tuple in for_combo2:
				predicted_combo.append(list((predicted_sales[for_combo.index(temp_tuple)],predicted_returns[for_combo2.index(temp_tuple)])))
				ground_truth_combo_method.append(ground_truth_sales_method[for_combo.index(temp_tuple)])
	
	predicted_combo = np.array(predicted_combo)
	balanced2 = (np.where(np.array(ground_truth_combo_method) == 0)[0])
	balanced3 = (np.where(np.array(ground_truth_combo_method) != 0)[0])
	balanced4 = np.random.choice(balanced2,balanced3.size,replace=False)
	balanced2 = np.concatenate([balanced3,balanced4])
	balanced3 = list(np.take(np.array(ground_truth_combo_method), balanced2))
	balanced4 = predicted_combo[balanced2[:,None],[0,1]]
	ground_truth_combo_method = balanced3
	predicted_combo = balanced4
	
	balanced2 = (np.where(np.array(ground_truth_sales_method) == 0)[0])
	balanced3 = (np.where(np.array(ground_truth_sales_method) != 0)[0])
	balanced4 = np.random.choice(balanced2,balanced3.size,replace=False)
	balanced2 = np.concatenate([balanced3,balanced4])
	balanced3 = list(np.take(np.array(ground_truth_sales_method), balanced2))
	balanced4 = list(np.take(np.array(predicted_sales), balanced2))
	ground_truth_sales_method = balanced3
	predicted_sales = balanced4
	
	print ((np.where(np.array(ground_truth_sales_method) == 0)[0]).size)
	print ((np.where(np.array(ground_truth_sales_method) != 0)[0]).size)
	print ((np.where(np.array(predicted_sales) == 0)[0]).size)
	print ((np.where(np.array(predicted_sales) != 0)[0]).size)
	
	balanced2 = (np.where(np.array(ground_truth_returns_method) == 0)[0])
	balanced3 = (np.where(np.array(ground_truth_returns_method) != 0)[0])
	balanced4 = np.random.choice(balanced2,balanced3.size,replace=False)
	balanced2 = np.concatenate([balanced3,balanced4])
	balanced3 = list(np.take(np.array(ground_truth_returns_method), balanced2))
	balanced4 = list(np.take(np.array(predicted_returns), balanced2))
	ground_truth_returns_method = balanced3
	predicted_returns = balanced4
	
	print ((np.where(np.array(ground_truth_returns_method) == 0)[0]).size)
	print ((np.where(np.array(ground_truth_returns_method) != 0)[0]).size)
	print ((np.where(np.array(predicted_returns) == 0)[0]).size)
	print ((np.where(np.array(predicted_returns) != 0)[0]).size)
	
	sales_model = svm.SVC()
	predicted_sales = np.asarray(predicted_sales)
	ground_truth_sales_method = np.asarray(ground_truth_sales_method)
	predicted_sales = predicted_sales.reshape(-1,1)
	sales_model.fit(predicted_sales,ground_truth_sales_method)
	if os.path.exists("../../data/"+str(vertical)+"/results/method_4/sales_model.pickle"):
		os.remove("../../data/"+str(vertical)+"/results/method_4/sales_model.pickle")
	with open("../../data/"+str(vertical)+"/results/method_4/sales_model.pickle",'w') as f:
		pickle.dump(sales_model,f)
	
	returns_model =  svm.SVC()
	predicted_returns = np.asarray(predicted_returns)
	ground_truth_returns_method = np.asarray(ground_truth_returns_method)
	predicted_returns = predicted_returns.reshape(-1,1)
	returns_model.fit(predicted_returns,ground_truth_returns_method)
	if os.path.exists("../../data/"+str(vertical)+"/results/method_4/returns_model.pickle"):
		os.remove("../../data/"+str(vertical)+"/results/method_4/returns_model.pickle")
	with open("../../data/"+str(vertical)+"/results/method_4/returns_model.pickle",'w') as f:
		pickle.dump(returns_model,f)
	
	combo_model = svm.SVC()
	predicted_combo = np.asarray(predicted_combo)
	ground_truth_combo_method = np.asarray(ground_truth_combo_method)
	combo_model.fit(predicted_combo,ground_truth_combo_method)
	if os.path.exists("../../data/"+str(vertical)+"/results/method_4/combo_model.pickle"):
		os.remove("../../data/"+str(vertical)+"/results/method_4/combo_model.pickle")
	with open("../../data/"+str(vertical)+"/results/method_4/combo_model.pickle",'w') as f:
		pickle.dump(combo_model,f)
	
	print "Returns Done."
	print "Models Learned."
	print "Testing Begins."
	
	predicted_sales = []
	predicted_returns = []
	predicted_combo = []
	for_combo = []
	for_combo2 = []
	ground_truth_sales_method = []
	ground_truth_returns_method = []
	ground_truth_combo_method = []
	
	ground_truth_returns = []
	
	for df in pd.read_csv(returns_testing_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			ground_truth_returns.append((rows['order_item_id'],rows['brand'],rows['account_id'],rows[size_column_name]))
	
	for df in pd.read_csv(sales_testing_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			
			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in sales_bs and (str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower() in cluster_sales_flat:
				clus = [x for x in cluster_sales if sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in x][0]
			else:
				continue
			
			if not G_sales_bi_weighted.has_node(str(rows['account_id'])):
				continue
			
			if G_returns_bi_weighted.has_node(str(rows['account_id'])):
				numerator = 0
				denominator = 0
				running_sum = 0
				total_sold = 0
				for items in clus:
					if G_returns_bi_weighted.has_edge(str(rows['account_id']),sales_bs[items]):
						numerator = G_returns_bi_weighted[str(rows['account_id'])][sales_bs[items]]['weight']
						denominator = numerator
					else:
						numerator = 0
					if G_sales_bi_weighted.has_edge(str(rows['account_id']),sales_bs[items]):
						denominator += G_sales_bi_weighted[str(rows['account_id'])][sales_bs[items]]['weight']
					elif denominator == 0:
						continue
					running_sum = float(running_sum) + ((float(numerator))/float(denominator))
					if numerator > 0:
						total_sold += 1
				if total_sold > 0:
					predicted_sales.append((float(running_sum))/float(total_sold))
					for_combo.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
				else:
					predicted_sales.append(0)
					for_combo.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			else:
				predicted_sales.append(0)
				for_combo.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			
			if (rows['order_id'],rows['brand'],rows['account_id'],rows[size_column_name]) in ground_truth_returns:
				ground_truth_sales_method.append(1)
			else:
				ground_truth_sales_method.append(0)
	
	print "Sales Predictions Done."
	
	for df in pd.read_csv(sales_testing_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			
			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in returns_bs and (str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower() in cluster_returns_flat:
				clus = [x for x in cluster_returns if returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in x][0]
			else:
				continue
			
			if not G_sales_bi_weighted.has_node(str(rows['account_id'])):
				continue
			
			if G_returns_bi_weighted.has_node(str(rows['account_id'])):
				numerator = 0
				denominator = 0
				running_sum = 0
				total_sold = 0
				for items in clus:
					if G_returns_bi_weighted.has_edge(str(rows['account_id']),returns_bs[items]):
						numerator = G_returns_bi_weighted[str(rows['account_id'])][returns_bs[items]]['weight']
						denominator = numerator
					else:
						numerator = 0
					if G_sales_bi_weighted.has_edge(str(rows['account_id']),returns_bs[items]):
						denominator += G_sales_bi_weighted[str(rows['account_id'])][returns_bs[items]]['weight']
					elif denominator == 0:
						continue
					running_sum = float(running_sum) + ((float(numerator))/float(denominator))
					if numerator > 0:
						total_sold += 1
				if total_sold > 0:
					predicted_returns.append((float(running_sum))/float(total_sold))
					for_combo2.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
				else:
					predicted_returns.append(0)
					for_combo2.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			else:
				predicted_returns.append(0)
				for_combo2.append((str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id'])))
			
			if (rows['order_id'],rows['brand'],rows['account_id'],rows[size_column_name]) in ground_truth_returns:
				ground_truth_returns_method.append(1)
			else:
				ground_truth_returns_method.append(0)
	
	for df in pd.read_csv(sales_testing_name, sep="\t",chunksize=chunksize):
		for index,rows in df.iterrows():
			if str(rows['brand']).lower() == 'null' or str(rows[size_column_name]).lower() == 'null':
				continue
			
			temp_tuple = (str(rows['account_id']),str(rows['brand']).lower(),str(rows[size_column_name]).lower(),str(rows['order_id']))
			if temp_tuple in for_combo and temp_tuple in for_combo2:
				predicted_combo.append(list((predicted_sales[for_combo.index(temp_tuple)],predicted_returns[for_combo2.index(temp_tuple)])))
				ground_truth_combo_method.append(ground_truth_sales_method[for_combo.index(temp_tuple)])
	
	if os.path.exists("../../data/"+str(vertical)+"/results/method_4/eval.txt"):
		os.remove("../../data/"+str(vertical)+"/results/method_4/eval.txt")
	eval_file = open("../../data/"+str(vertical)+"/results/method_4/eval.txt","w")
	
	predicted_sales = np.asarray(predicted_sales)
	predicted_sales = predicted_sales.reshape(-1,1)
	temp = sales_model.predict(predicted_sales)
	ground_truth_sales_method = np.asarray(ground_truth_sales_method)
	print "For Sales only method."
	print "Accuracy:"+str(accuracy_score(ground_truth_sales_method,temp))
	print "Precision:"+str(precision_score(ground_truth_sales_method,temp))
	print "Recall:"+str(recall_score(ground_truth_sales_method,temp))
	print "F1 Score:"+str(f1_score(ground_truth_sales_method,temp))
	print "ROC AUC:"+str(roc_auc_score(ground_truth_sales_method,temp))
	print ("\n")
	
	eval_file.write (str(accuracy_score(ground_truth_sales_method,temp))+"\n")
	eval_file.write (str(precision_score(ground_truth_sales_method,temp))+"\n")
	eval_file.write (str(recall_score(ground_truth_sales_method,temp))+"\n")
	eval_file.write (str(f1_score(ground_truth_sales_method,temp))+"\n")
	eval_file.write (str(roc_auc_score(ground_truth_sales_method,temp))+"\n")
	eval_file.write ("\n")
	
	predicted_returns = np.asarray(predicted_returns)
	predicted_returns = predicted_returns.reshape(-1,1)
	temp = returns_model.predict(predicted_returns)
	ground_truth_returns_method = np.asarray(ground_truth_returns_method)
	print "For Returns only method."
	print "Accuracy:"+str(accuracy_score(ground_truth_returns_method,temp))
	print "Precision:"+str(precision_score(ground_truth_returns_method,temp))
	print "Recall:"+str(recall_score(ground_truth_returns_method,temp))
	print "F1 Score:"+str(f1_score(ground_truth_returns_method,temp))
	print "ROC AUC:"+str(roc_auc_score(ground_truth_returns_method,temp))
	print ("\n")
	
	eval_file.write (str(accuracy_score(ground_truth_returns_method,temp))+"\n")
	eval_file.write (str(precision_score(ground_truth_returns_method,temp))+"\n")
	eval_file.write (str(recall_score(ground_truth_returns_method,temp))+"\n")
	eval_file.write (str(f1_score(ground_truth_returns_method,temp))+"\n")
	eval_file.write (str(roc_auc_score(ground_truth_returns_method,temp))+"\n")
	eval_file.write ("\n")
	
	predicted_combo = np.asarray(predicted_combo)
	temp = combo_model.predict(predicted_combo)
	ground_truth_combo_method = np.asarray(ground_truth_combo_method)
	print "For Combo method."
	print "Accuracy:"+str(accuracy_score(ground_truth_combo_method,temp))
	print "Precision:"+str(precision_score(ground_truth_combo_method,temp))
	print "Recall:"+str(recall_score(ground_truth_combo_method,temp))
	print "F1 Score:"+str(f1_score(ground_truth_combo_method,temp))
	print "ROC AUC:"+str(roc_auc_score(ground_truth_combo_method,temp))
	
	eval_file.write (str(accuracy_score(ground_truth_combo_method,temp))+"\n")
	eval_file.write (str(precision_score(ground_truth_combo_method,temp))+"\n")
	eval_file.write (str(recall_score(ground_truth_combo_method,temp))+"\n")
	eval_file.write (str(f1_score(ground_truth_combo_method,temp))+"\n")
	eval_file.write (str(roc_auc_score(ground_truth_combo_method,temp))+"\n")
	eval_file.write ("\n\n\n")
	
	eval_file.write (str(float(not_clustered_sales)/total_not_null_sales)+"\n")
	eval_file.write (str(float(user_cold_start_sales)/total_not_null_sales)+"\n")
	eval_file.write ("\n")
	
	eval_file.write (str(float(not_clustered_returns)/total_not_null_returns)+"\n")
	eval_file.write (str(float(user_cold_start_returns)/total_not_null_returns)+"\n")
	eval_file.write ("\n")
	
	eval_file.close()

def clustering_init (vertical):
	if str(vertical) == "womenjean":
		k = 7
		print "For "+str(vertical)+" at k = "+str(k)+"."
		clustering(vertical,k)
		evaluation(vertical)
		os.system("mv ../../data/"+str(vertical)+"/results/method_4/eval.txt ../../data/"+str(vertical)+"/results/method_4/eval_"+str(k)+".txt ")
	else:
		k = 5
		print "For "+str(vertical)+" at k = "+str(k)+"."
		clustering(vertical,k)
		evaluation(vertical)
		os.system("mv ../../data/"+str(vertical)+"/results/method_4/eval.txt ../../data/"+str(vertical)+"/results/method_4/eval_"+str(k)+".txt ")

def main():
	verticals = ["menpolotshirt"]
	for vertical in verticals:
		clustering_init(vertical)

if __name__ == '__main__':
	main()
