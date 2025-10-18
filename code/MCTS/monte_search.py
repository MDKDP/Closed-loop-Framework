import pickle
from tqdm import tqdm
import numpy as np
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def search_max_path(start_node, G):
    target_node = 248
    max_path_length = 5
    num_samples = 100000
    max_path_probability = 0
    max_path = []

    for _ in range(num_samples):
        current_node = start_node
        path_probability = 1.0
        path_length = 0
        node_record = [current_node]

        while current_node != target_node and path_length <= max_path_length:
            neighbors = list(G.successors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            edge_weight = G[current_node][next_node]['weight']
            path_probability *= edge_weight
            current_node = next_node
            path_length += 1
            node_record.append(next_node)

        if current_node == target_node and path_length <= max_path_length:
            if path_probability > max_path_probability:
                max_path_probability = path_probability
                max_path = node_record

    return [start_node, max_path_probability, max_path]


def monte_search(save_flag, kg_refinement, training_epoch, train_time, seed):
    random.seed(seed)
    if not kg_refinement:
        g_path = "../../results/RGHAT/initial_result/{0}".format(train_time)
    else:
        g_path = "../../results/RGHAT/training_result/{0}/training_epoch_{1}".format(train_time, training_epoch)
    G = np.load(os.path.join(g_path, 'G_network_sqrt.pkl'), allow_pickle=True)
    start_nodes = list(range(249))
    G_copy = [G for _ in range(249)]
    results = []
    print("Monte searching")
    # parallel processing
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(search_max_path, node, G) for node, G in zip(start_nodes, G_copy)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)
            print(f"start node: {result[0]} -> end node: 248, transition probability: {result[1]}")
            print("path: ", result[2])

    results = sorted(results)
    record_max_pro = [i[1] for i in results]
    record_max_path = [i[2] for i in results]
    # save the transition probability and path
    if save_flag:
        if not kg_refinement:
            save_dir = "../../results/RGHAT/initial_result/{0}".format(train_time)
        else:
            save_dir = "../../results/RGHAT/training_result/{0}/training_epoch_{1}".format(train_time, training_epoch)
        with open(os.path.join(save_dir, "record_max_pro_sqrt.pkl"), 'wb') as f1:
            pickle.dump(record_max_pro, f1)
        with open(os.path.join(save_dir, "record_max_path_sqrt.pkl"), 'wb') as f2:
            pickle.dump(record_max_path, f2)


if __name__ == '__main__':
    save_flag, kg_refinement = True, False
    monte_search(save_flag, kg_refinement, training_epoch=None, train_time=None, seed=42)
