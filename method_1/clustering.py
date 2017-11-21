import pickle
import networkx as nx
import pandas as pd
import numpy as np
import random
import sys
import subprocess
import os
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def clustering(vertical):
	
	for both in ["sales","returns"]:
		print "Starting Clustering of "+str(vertical)+" "+str(both)+" with Method-1."
		
		G=nx.read_gpickle("../../data/"+str(vertical)+"/"+str(both)+"/pickles/"+str(vertical)+"_"+str(both)+"_training_pro.pickle")
		
		os.system("sed -e \"s/ /\\t/g\" ../../data/"+str(vertical)+"/"+str(both)+"/data/"+str(vertical)+"_"+str(both)+"_training_edgelist.txt > ../../data/"+str(vertical)+"/"+str(both)+"/data/temp")
		os.system("rm ../../data/"+str(vertical)+"/"+str(both)+"/data/"+str(vertical)+"_"+str(both)+"_training_edgelist.txt")
		os.system("mv ../../data/"+str(vertical)+"/"+str(both)+"/data/temp "+"../../data/"+str(vertical)+"/"+str(both)+"/data/"+str(vertical)+"_"+str(both)+"_training_edgelist.txt")
		
		content = []
		
		with open("../../data/"+str(vertical)+"/"+str(both)+"/data/"+str(vertical)+"_"+str(both)+"_training_edgelist.txt") as f:
			content = f.readlines()
		
		vertices = set()
		
		clusters = {}
		
		for lines in content:
			temp = map(int,lines[:-1].split("\t"))
			vertices.add(temp[0])
			vertices.add(temp[1])
			clusters[temp[0]] = []
			clusters[temp[1]] = []
		
		vertices = list(vertices)
		
		graphs = list(nx.connected_component_subgraphs(G))
		
		d = 1
		for gs in graphs:
			d = max(d,nx.diameter(gs))
		
		cr = (1-float(0.045**(1/float(d))))
		
		print "cr calculated."
		
		for seed in vertices:
			os.system("echo "+str(seed)+" > seed.txt")
			most_similar = subprocess.check_output(["python", "lib/run_walker.py" , "../../data/"+str(vertical)+"/"+str(both)+"/data/"+str(vertical)+"_"+str(both)+"_training_edgelist.txt", "seed.txt"])
			
			most_similar = most_similar.split("\n")[1:-1]
			
			os.system("rm seed.txt")
			
			for lines in most_similar:
				if float(lines.split("\t")[1]) > 0.0:
					clusters[seed].append((int(lines.split("\t")[0]),float(lines.split("\t")[1])))
				else:
					break
			
			if len(clusters[seed]) == 0:
				clusters.pop(seed)
				continue
			
			mean = float(float(sum(x[1] for x in clusters[seed]))/float(len(clusters[seed])))
			temp_list = [i[0] for i in clusters[seed] if i[1] >= mean]
			clusters[seed] = temp_list[:50]
		
		print str(len(clusters.keys()))+" BSs were clustered."
		
		os.system("mkdir ../../data/"+str(vertical)+"/results/method_1/")
		
		if os.path.exists("../../data/"+str(vertical)+"/results/method_1/"+str(both)+"_clusters.pickle"):
			os.remove("../../data/"+str(vertical)+"/results/method_1/"+str(both)+"_clusters.pickle")
		
		with open("../../data/"+str(vertical)+"/results/method_1/"+str(both)+"_clusters.pickle",'w') as f:
			pickle.dump(clusters,f)

def evaluation(vertical):
	
	print "Training of "+str(vertical)+" begins."
	
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
	
	cluster_sales = pickle.load(open("../../data/"+str(vertical)+"/results/method_1/sales_clusters.pickle","r"))
	cluster_returns = pickle.load(open("../../data/"+str(vertical)+"/results/method_1/returns_clusters.pickle","r"))
	sales_bs = list(pickle.load(open("../../data/"+str(vertical)+"/sales/pickles/"+str(vertical)+"_sales_training_bs.pickle","r")))
	sales_bs.sort()
	returns_bs = list(pickle.load(open("../../data/"+str(vertical)+"/returns/pickles/"+str(vertical)+"_returns_training_bs.pickle","r")))
	returns_bs.sort()
	
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
			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in sales_bs and sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in cluster_sales.keys() :
				clus = list(cluster_sales[sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())])
				clus.append(sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()))
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
			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in returns_bs and returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in cluster_returns.keys():
				clus = list(cluster_returns[returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())])
				clus.append(returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()))
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
	if os.path.exists("../../data/"+str(vertical)+"/results/method_1/sales_model.pickle"):
		os.remove("../../data/"+str(vertical)+"/results/method_1/sales_model.pickle")
	with open("../../data/"+str(vertical)+"/results/method_1/sales_model.pickle",'w') as f:
		pickle.dump(sales_model,f)
	
	returns_model =  svm.SVC()
	predicted_returns = np.asarray(predicted_returns)
	ground_truth_returns_method = np.asarray(ground_truth_returns_method)
	predicted_returns = predicted_returns.reshape(-1,1)
	returns_model.fit(predicted_returns,ground_truth_returns_method)
	if os.path.exists("../../data/"+str(vertical)+"/results/method_1/returns_model.pickle"):
		os.remove("../../data/"+str(vertical)+"/results/method_1/returns_model.pickle")
	with open("../../data/"+str(vertical)+"/results/method_1/returns_model.pickle",'w') as f:
		pickle.dump(returns_model,f)

	combo_model = svm.SVC()
	predicted_combo = np.asarray(predicted_combo)
	ground_truth_combo_method = np.asarray(ground_truth_combo_method)
	combo_model.fit(predicted_combo,ground_truth_combo_method)
	if os.path.exists("../../data/"+str(vertical)+"/results/method_1/combo_model.pickle"):
		os.remove("../../data/"+str(vertical)+"/results/method_1/combo_model.pickle")
	with open("../../data/"+str(vertical)+"/results/method_1/combo_model.pickle",'w') as f:
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

			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in sales_bs and sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in cluster_sales.keys():
				clus = list(cluster_sales[sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())])
				clus.append(sales_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()))
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
			
			if ((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in returns_bs and returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()) in cluster_returns.keys():
				clus = list(cluster_returns[returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower())])
				clus.append(returns_bs.index((str(rows['brand']).lower()).replace(" ","_")+";"+str(rows[size_column_name]).lower()))
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
	
	if os.path.exists("../../data/"+str(vertical)+"/results/method_1/eval.txt"):
		os.remove("../../data/"+str(vertical)+"/results/method_1/eval.txt")
	eval_file = open("../../data/"+str(vertical)+"/results/method_1/eval.txt","w") 
	
	predicted_sales = np.asarray(predicted_sales)
	predicted_sales = predicted_sales.reshape(-1,1)
	temp = sales_model.predict(predicted_sales)
	ground_truth_sales_method = np.asarray(ground_truth_sales_method)
	print ("For Sales only method.")
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
	print ("For Returns only method.")
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
	print ("For Combo method.")
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


def main():
	verticals = ["womenjean"]
	for vertical in verticals:
		clustering(vertical)
		evaluation(vertical)

if __name__ == '__main__':
	main()
