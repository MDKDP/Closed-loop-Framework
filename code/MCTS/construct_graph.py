import numpy as np
import pickle
import math
import networkx as nx
import os
from scipy import sparse


def construct_graph(save_flag, kg_refinement, training_epoch, train_time):
    # load gamma
    if not kg_refinement:
        data_dir = "../../results/JDeC/initial_result/{0}".format(train_time)
    else:
        data_dir = "../../results/JDeC/training_result/{0}/training_epoch_{1}".format(train_time, training_epoch)
    all_b_hrt = np.load(os.path.join(data_dir, 'all_b_hrt.pkl'), allow_pickle=True)
    all_b_hrt = [sparse.coo_matrix(i.cpu().detach().numpy()) for i in all_b_hrt]

    # load entity count
    graph_nodes_count = np.load('../../datasets/data_g/graph_nodes_count.pkl', allow_pickle=True)
    graph_nodes_weight = {k: np.log(v + 1) for k, v in graph_nodes_count.items()}

    # calculate the transition score for each triplet
    for i in range(len(all_b_hrt)):
        all_b_hrt[i] = all_b_hrt[i] * math.sqrt(all_b_hrt[i].col.shape[0])
        all_b_hrt[i] = sparse.coo_matrix((np.exp(all_b_hrt[i].data), (all_b_hrt[i].row, all_b_hrt[i].col)), shape=all_b_hrt[i].shape)
    headLst, relationLst, tileLst, weightLst = [], [], [], []
    for headid in range(len(all_b_hrt)):
        relationLst += list(all_b_hrt[headid].row)
        tail = all_b_hrt[headid].col
        tileLst += list(tail)
        node_weight = np.array([graph_nodes_weight[t] for t in tail])
        weightLst += list(all_b_hrt[headid].data * node_weight)
        headLst += [headid] * len(list(all_b_hrt[headid].data))
    # normalization
    weightLst = np.array(weightLst)
    max_weight = max(weightLst)
    min_weight = min(weightLst)
    weightLst = list((weightLst - min_weight) / (max_weight - min_weight))

    # construct the knowledge graph
    G = nx.DiGraph()
    G.add_nodes_from(range(248))
    for i in range(len(headLst)):
        if headLst[i] in [1, 155, 180, 202, 247]:  # if subcategory, add directed edge
            G.add_edge(tileLst[i], headLst[i], weight=weightLst[i])
        else:
            G.add_edge(headLst[i], tileLst[i], weight=weightLst[i])

    # add 5 class node to depression
    # category weight: physical symptom 1 ; psychological symptom 1 ; life event 1 ; medication 2 ; therapy 2
    for idx, h in enumerate([1, 155, 180, 202, 247]):
        if h == 180 or h == 247:
            G.add_edge(h, 248, weight=2)
        else:
            G.add_edge(h, 248, weight=1)
    print(G)

    if save_flag:
        if not kg_refinement:
            save_dir = "../../results/RGHAT/initial_result/{0}".format(train_time)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = "../../results/RGHAT/training_result/{0}/training_epoch_{1}".format(train_time, training_epoch)
            os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "G_network_sqrt.pkl"), 'wb') as f5:
            pickle.dump(G, f5)


if __name__ == '__main__':
    save_flag, kg_refinement = True, False
    construct_graph(save_flag, kg_refinement, training_epoch=None, train_time=None)
