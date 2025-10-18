import numpy as np
import torch
from .load_data_rghat import Data
import scipy.sparse as sp
import random as rd
import collections


class RGHAT_loader(Data):
    def __init__(self, args, train_time, train_epoch, kg_refinement, kg_expansion, kg_pretrain):
        super().__init__(args, train_time, train_epoch, kg_refinement, kg_expansion, kg_pretrain)

        # generate the sparse adjacency matrices for user-item interaction & relational kg data_network.
        self.train_adj_list, self.train_adj_r_list = self._get_relational_adj_list(self.train_relation_dict)
        self.test_adj_list, self.test_adj_r_list = self._get_relational_adj_list(self.test_relation_dict)

        # generate the triples dictionary, key is 'head', value is '(relation, tail)'.
        self.train_all_kg_dict, self.train_all_relation_dict = self._get_all_kg_dict(self.train_adj_list, self.train_adj_r_list)
        self.test_all_kg_dict, self.test_all_relation_dict = self._get_all_kg_dict(self.test_adj_list, self.test_adj_r_list)

        self.hr_norm = self._get_hr_matrix()
        self.hrt_norm = self._get_hrv_matrix()

        self.train_all_h_list, self.train_all_r_list, self.train_all_t_list = self._get_all_kg_data(self.train_adj_list, self.train_adj_r_list)  # h, r, t, v 按照h, t升序排列
        self.test_all_h_list, self.test_all_r_list, self.test_all_t_list = self._get_all_kg_data(self.test_adj_list, self.test_adj_r_list)

    def _get_relational_adj_list(self, relation_dict):
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat):
            n_all = self.n_entities
            sp_rows = np_mat[:, 0]
            sp_cols = np_mat[:, 1]
            sp_vals = [1.0] * np_mat.shape[0]
            sp_adj = sp.coo_matrix((sp_vals, (sp_rows, sp_cols)), shape=(n_all, n_all))

            return sp_adj

        for r_id in sorted(relation_dict.keys()):
            sp_adj = _np_mat2sp_adj(np.array(relation_dict[r_id]))
            adj_mat_list.append(sp_adj)
            adj_r_list.append(r_id)

        return adj_mat_list, adj_r_list

    def _get_all_kg_dict(self, adj_list, adj_r_list):
        all_kg_dict = collections.defaultdict(list)  # head:[(rel,tile)]
        all_relation_dict = collections.defaultdict(list)  # head:[rel]
        for l_id, lap in enumerate(adj_list):

            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = adj_r_list[l_id]

                all_kg_dict[head].append((relation, tail))
                all_relation_dict[head].append(relation)
        for k, v in all_relation_dict.items():
            all_relation_dict[k] = list(set(v))

        return all_kg_dict, all_relation_dict

    def _get_hr_matrix(self):
        hr_matrix = np.zeros((self.n_entities, self.n_relations))
        for head_idx in self.train_all_relation_dict.keys():
            r_lst = self.train_all_relation_dict[head_idx]
            h_r_score = [1.0 / len(r_lst)] * len(r_lst)
            for r_i, h_r_i in zip(r_lst, h_r_score):
                hr_matrix[head_idx][r_i] = h_r_i

        return hr_matrix

    def _get_hrv_matrix(self):
        b_hrt = []
        for head_idx in range(self.n_entities):
            if head_idx in self.train_all_kg_dict.keys():
                rt_matrix = np.zeros((self.n_relations, self.n_entities))
                tripet = self.train_all_kg_dict[head_idx]
                for i in tripet:
                    rt_matrix[i[0]][i[1]] = 1.0
                row_sums = rt_matrix.sum(axis=1, keepdims=True)
                mask = row_sums != 0
                mask = mask.flatten()
                rt_matrix[mask] = rt_matrix[mask] / row_sums[mask]
                hr_matrix_row = self.hr_norm[head_idx].reshape([-1, 1])
                hrt_matrix = rt_matrix * hr_matrix_row
                b_hrt.append(hrt_matrix)
            else:
                rt_matrix = np.zeros((self.n_relations, self.n_entities))
                b_hrt.append(rt_matrix)
        return b_hrt

    def _get_all_kg_data(self, adj_list, adj_r_list):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []

        for l_id, lap in enumerate(adj_list):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_r_list += [adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in adj_list])
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[], []]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list]

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list = [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])

        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)

        return new_h_list, new_r_list, new_t_list

    def _generate_train_A_batch(self):
        exist_heads = self.train_all_kg_dict.keys()
        # print(len(exist_heads))
        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.train_all_kg_dict[h]
            # print(h, pos_triples)
            n_pos_triples = len(pos_triples)
            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
                r = pos_triples[pos_id][0]
                t = pos_triples[pos_id][1]
                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break
                t = np.random.randint(low=0, high=self.n_entities, size=1)[0]
                if self.kg_refinement:
                    if (r, t) not in self.train_all_kg_dict[h] and tuple([h, r, t]) not in neg_dict and t not in neg_ts:
                        neg_ts.append(t)
                else:
                    if (r, t) not in self.train_all_kg_dict[h] and tuple([h, r, t]) and t not in neg_ts:
                        neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        if self.kg_refinement:
            if len(self.kg_neg_with_weight) == 0:
                combined = np.array(self.kg_neg_without_weight)
            elif len(self.kg_neg_without_weight) == 0:
                combined = np.array(self.kg_neg_with_weight)
            else:
                combined = np.concatenate((self.kg_neg_with_weight, self.kg_neg_without_weight))
            neg_dict = [tuple(i) for i in combined]
        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts
            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 2)
            neg_t_batch += [neg_ts]

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_A_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_A_batch()
        batch_data = {}
        batch_data['heads'] = torch.tensor(heads)
        batch_data['relations'] = torch.tensor(relations)
        batch_data['pos_tails'] = torch.tensor(pos_tails)
        batch_data['neg_tails'] = torch.tensor(neg_tails)
        return batch_data

    def generate_train_A_feed_dict(self, model, batch_data):
        feed_dict = {
            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails'],
        }

        return feed_dict

    def generate_test_feed_dict(self, model, h, r, t):

        feed_dict = {
            model.h: h,
            model.r: r,
            model.pos_t: t
        }

        return feed_dict
