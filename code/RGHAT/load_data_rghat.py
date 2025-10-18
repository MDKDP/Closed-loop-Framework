import collections
import numpy as np
import pandas as pd


class Data(object):
    def __init__(self, args, train_time, train_epoch, kg_refinement, kg_expansion, kg_pretrain):
        self.args = args
        self.train_time = train_time
        self.train_epoch = train_epoch
        self.kg_refinement = kg_refinement
        self.kg_expansion = kg_expansion
        self.kg_pretrain = kg_pretrain
        self.batch_size_kg = args.batch_size_kg
        self.n_relations, self.n_entities = 0, 0
        if self.kg_expansion or self.kg_pretrain:
            self.KG = self.load_triplet_of_expansion_or_pretrain()
            self.train_kg_dict, self.train_relation_dict, self.test_kg_dict, self.test_relation_dict = self.load_kg_of_expansion_or_pretrain()
        if self.kg_refinement:
            self.KG = self.load_triplet_of_refinement()
            self.train_kg_dict, self.train_relation_dict, self.test_kg_dict, self.test_relation_dict = self.load_kg_of_refinement()

    def load_triplet_of_refinement(self):
        initial_kg = np.array(pd.read_csv('../../datasets/data_g/kg_final_full.csv'))

        # Newly identified triples from depression patients' post
        pos_triplets = pd.read_csv('../../results/RGHAT/training_result/{0}/training_epoch_{1}/detected_with_depression_entities.csv'.format(self.train_time, self.train_epoch))
        p_triplets = np.array([[int(x) for x in i.strip("[]").split(', ')] for i in np.array(pos_triplets)[:, 0]])
        initial_kg_set = {tuple(r[:-1]) for r in initial_kg}
        p_triplets_in_initial_kg = np.array([tuple(row) in initial_kg_set for row in p_triplets])
        is_new_flag = np.where(p_triplets_in_initial_kg, 'Old knowledge', 'New knowledge')
        pos_triplets['New_or_Old'] = is_new_flag
        pos_triplets.to_csv('../../results/RGHAT/training_result/{0}/training_epoch_{1}/detected_with_depression_entities.csv'.format(self.train_time, self.train_epoch), index=False)

        # Combine the newly identified triples from depression patients' post with the current knowledge graph triples.
        kg_counts = collections.defaultdict(int)
        for row in initial_kg:
            h, r, t, c = row
            kg_counts[(h, r, t)] += c
        for row in p_triplets:
            h, r, t = row
            if r != 15:
                kg_counts[(h, r, t)] += 1
                kg_counts[(t, r, h)] += 1
            else:
                print(row)

        initial_kg = np.array([list(k) + [v] for k, v in kg_counts.items()])
        self.n_relations = max(initial_kg[:, 1]) + 1
        self.n_entities = max(max(initial_kg[:, 0]), max(initial_kg[:, 2])) + 1

        return initial_kg

    def load_kg_of_refinement(self):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((relation, tail))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = self.KG[:, :-1]
        kg_np_counts = {tuple(i[:-1]): i[-1] for i in self.KG}

        # Newly identified triples from non-depression patients' post
        neg_triplets_df = pd.read_csv('../../results/RGHAT/training_result/{0}/training_epoch_{1}/detected_without_depression_entities.csv'.format(self.train_time, self.train_epoch))
        neg_triplet = np.array([[int(x) for x in i.strip("[]").split(', ')] for i in np.array(neg_triplets_df)[:, 0]])
        neg_triplet_reversed = neg_triplet[:, [2, 1, 0]]
        neg_triplets_df_reversed = neg_triplets_df.copy()
        neg_triplets_df_reversed['triplet'] = ['[{0}, {1}, {2}]'.format(i[0], i[1], i[2]) for i in neg_triplet_reversed]
        neg_triplets_df = pd.concat([neg_triplets_df, neg_triplets_df_reversed], axis=0)

        neg_triplet = np.concatenate((neg_triplet, neg_triplet_reversed))
        neg_triplet, neg_counts = np.unique(neg_triplet, axis=0, return_counts=True)

        # Recognize the false positive triplets
        initial_kg = np.array(pd.read_csv('../../datasets/data_g/kg_final_full.csv'))[:, :-1]
        kg_np_set = {tuple(row) for row in initial_kg}
        mask = [tuple(t) in kg_np_set for t in neg_triplet]
        false_positive_triplet = neg_triplet[mask]
        false_positive_counts = neg_counts[mask]
        false_positive_triplet_str = ['[{0}, {1}, {2}]'.format(i[0], i[1], i[2]) for i in false_positive_triplet]
        false_positive_flag = ['Yes' if i in false_positive_triplet_str else 'No' for i in neg_triplets_df['triplet']]
        neg_triplets_df['false positive flag'] = false_positive_flag
        neg_triplets_df.to_csv('../../results/RGHAT/training_result/{0}/training_epoch_{1}/detected_without_depression_entities.csv'.format(self.train_time, self.train_epoch), index=False)

        if len(false_positive_triplet) == 0:  # normal triplets
            kg_pos = kg_np
            kg_neg_with_weight = []
            kg_neg_weight = []
            kg_neg_without_weight = neg_triplet
            kg_pos_counts = np.array(list(kg_np_counts.values()))
        else:
            # kg_pos: webmd + with depression triplet
            kg_pos = kg_np
            kg_pos_counts = np.array(list(kg_np_counts.values()))

            # kg_neg: 1.false positive triplet(kg_neg_with_weight) 2.not false positive triplet(kg_neg_without_weight)
            fp_set = {tuple(row) for row in false_positive_triplet}
            kg_neg_with_weight = false_positive_triplet.copy()
            c_h_r_t = np.array([kg_np_counts[i] for i in fp_set])
            c_h_r_t_p = false_positive_counts
            kg_neg_weight = c_h_r_t_p / (c_h_r_t + c_h_r_t_p)

            mask2 = np.array([tuple(row) not in fp_set for row in neg_triplet])
            kg_neg_without_weight = neg_triplet[mask2]

        kg_np = kg_pos
        train_kg_np = kg_np
        test_idx = np.where(kg_pos_counts >= 5)[0]  # Select some frequently occurring triples for testing(Our goal is to get the whole kg embed)
        test_kg_np = kg_np[test_idx]
        train_kg_dict, train_relation_dict = _construct_kg(train_kg_np)
        test_kg_dict, test_relation_dict = _construct_kg(test_kg_np)
        self.kg_neg_with_weight = kg_neg_with_weight
        self.kg_neg_weight = kg_neg_weight
        self.kg_neg_without_weight = kg_neg_without_weight

        return train_kg_dict, train_relation_dict, test_kg_dict, test_relation_dict

    def load_triplet_of_expansion_or_pretrain(self):
        initial_kg = np.array(pd.read_csv('../../datasets/data_g/kg_final_full.csv'))
        self.n_relations = max(initial_kg[:, 1]) + 1
        self.n_entities = max(max(initial_kg[:, 0]), max(initial_kg[:, 2])) + 1
        return initial_kg

    def load_kg_of_expansion_or_pretrain(self):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)
            for head, relation, tail in kg_np:
                kg[head].append((relation, tail))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = self.KG[:, :-1]
        train_kg_np = kg_np
        test_idx = np.where(self.KG[:, -1] >= 5)[0]  # Select some frequently occurring triples for testing(Our goal is to get the whole kg embed)
        test_kg_np = kg_np[test_idx]
        train_kg_dict, train_relation_dict = _construct_kg(train_kg_np)
        test_kg_dict, test_relation_dict = _construct_kg(test_kg_np)

        return train_kg_dict, train_relation_dict, test_kg_dict, test_relation_dict
