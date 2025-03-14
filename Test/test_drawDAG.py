#to access files and folders
import os
#data analysis and manipulation library
import pandas as pd
from argparse import ArgumentParser

import warnings
warnings.simplefilter("ignore")#ignore warnings during executiona

import sys
sys.path.append('/kaggle/working/cogload/BN_CognitiveLoad/')
from ProcessingData import Processing
from BN import BN

#argument parser
parser = ArgumentParser()
parser.add_argument("--data_labels_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--data_path", default = "/kaggle/input/cognitiveload/Feature_selection/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
parser.add_argument("--method", default = 'hill_climbing', type = str, help = "Method to draw DAG (hill_climbing or tabu_search)")

args = parser.parse_args()

args_dict = vars(args)
log_args = pd.DataFrame([args_dict])

directory_name = '/kaggle/working/log/'
directory_result = '/kaggle/working/result/'
if not os.path.exists(directory_name):
    os.makedirs(directory_result)
    os.makedirs(directory_name)
file_name = f'args.csv'  
log_args.to_csv(os.path.join(directory_name, file_name), index=False)

#read the data
label_df = pd.read_excel(args.data_labels_path + 'labels.xlsx',index_col=0)
data = pd.read_csv(args.data_path + 'discrete_data.csv')
print("Data shapes:")
print('Labels',label_df.shape)
print('Data',data.shape)

#Processing data
process = Processing(data, label_df)
X_train, y_train, X_test, y_test, user_train, user_test = process.get_Data()

# Draw DAG
bn = BN(data, method=args.method)
bn.fit(X_train, y_train, user_train, args.GroupKFold)
accuracy = bn.predict(X_test, y_test)
print(f"Accuracy: {accuracy}")
pdt = bn.get_PDT()
for node, cpd in pdt.items():
    print(f"CPD của {node}:\n{cpd}\n")

import networkx as nx
import matplotlib.pyplot as plt

edges = bn.edges
G = nx.DiGraph()
G.add_edges_from(edges)

plt.figure(figsize=(12, 8))
pos = nx.nx_pydot.pydot_layout(G, prog="dot")  # Thay vì pygraphviz

nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue",
        font_size=10, font_weight="bold", edge_color="gray", arrows=True, 
        arrowsize=20, connectionstyle="arc3,rad=0.1")

plt.title("Directed Graph Representation")
plt.savefig("/kaggle/working/graph.png", dpi=300, bbox_inches="tight")  # Lưu với độ phân giải cao
plt.show()
