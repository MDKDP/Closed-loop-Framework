import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class RGHAT(nn.Module):
    def __init__(self, data_config, args):
        super(RGHAT, self).__init__()
        self.B_hrt = None
        self.hr_score = None
        self.entity_embed_after_encoder = None
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self._parse_args(data_config, args)
        self._build_weights()
        self.wc_h_r_t = None

    def _parse_args(self, data_config, args):
        self.kg_refinement = data_config['kg_refinement']
        if self.kg_refinement:
            self.kg_neg_with_weight = torch.tensor(data_config['kg_neg_with_weight']).to(self.device)
            self.kg_neg_weight = torch.tensor(data_config['kg_neg_weight'], dtype=torch.float32).to(self.device)
            self.kg_neg_without_weight = torch.tensor(data_config['kg_neg_without_weight']).to(self.device)

        self.emb_dim = args.embed_size
        self.b_hrt = [torch.tensor(i, dtype=torch.float32).to(self.device) for i in data_config['hrt_norm']]
        self.hr_matrix = torch.tensor(data_config['hr_norm'], dtype=torch.float32).to(self.device)
        self.pretrain_emb = torch.tensor(data_config['pretrain_emb'], dtype=torch.float32)
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']
        self.train_all_h_list = data_config['train_all_h_list']
        self.train_all_r_list = data_config['train_all_r_list']
        self.train_all_t_list = data_config['train_all_t_list']
        self.train_all_kg_dict = data_config['train_all_kg_dict']
        self.train_all_relation_dict = data_config['train_all_relation_dict']
        self.test_all_kg_dict = data_config['test_all_kg_dict']
        self.test_all_relation_dict = data_config['test_all_relation_dict']
        self.lr = args.lr
        self.kge_dim = args.kge_size
        self.batch_size_kg = args.batch_size_kg
        self.regs = args.regs
        self.temperature = args.temperature

    def _build_weights(self):
        self.entity_embed = nn.Embedding.from_pretrained(self.pretrain_emb, freeze=False)  # 重写
        self.relation_embed = nn.Embedding(self.n_relations, self.emb_dim)  # 重写
        nn.init.xavier_normal_(self.relation_embed.weight)

        self.W1 = nn.Linear(2 * self.emb_dim, self.emb_dim, bias=False)
        self.W2 = nn.Linear(2 * self.emb_dim, self.emb_dim, bias=False)
        self.W3 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W4 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.Q = nn.Linear(44160, self.emb_dim, bias=False)
        self.p = nn.Parameter(torch.randn(self.emb_dim))
        self.q = nn.Parameter(torch.randn(self.emb_dim))

        self.conv2d = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, bias=False)

        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)
        nn.init.xavier_normal_(self.W3.weight)
        nn.init.xavier_normal_(self.W4.weight)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.p.unsqueeze(0))
        nn.init.xavier_normal_(self.q.unsqueeze(0))

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        h_e = self.entity_embed_after_encoder[h]
        pos_t_e = self.entity_embed_after_encoder[pos_t]
        neg_t_e = self.entity_embed_after_encoder[neg_t]
        r_e = self.relation_embed.weight[r]
        return h_e, r_e, pos_t_e, neg_t_e

    def _get_kg_inference_for_neg(self, h, r, t):
        h_e = self.entity_embed_after_encoder[h]
        r_e = self.relation_embed.weight[r]
        t_e = self.entity_embed_after_encoder[t]

        return h_e, r_e, t_e

    def _create_bi_interaction_embed(self):
        B_hrt = self.B_hrt
        rows, cols, embs = [], [], []
        for b in B_hrt:
            rows.append(b[0][0])  # src
            cols.append(b[0][-1])  # tgt
            embs.append(b[1])  # b_hrt embedding

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.stack(embs)

        b_hrt_sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.n_entities, self.n_entities, self.emb_dim)
        )

        hrt_weights = torch.stack([hrt.sum(dim=0) for hrt in self.b_hrt])
        hrt_weights = hrt_weights.unsqueeze(-1)

        side_embedding = torch.bmm(
            hrt_weights.transpose(1, 2),
            b_hrt_sparse.to_dense()
        ).squeeze(1)
        ori_embedding = self.entity_embed.weight
        add_embedding = ori_embedding + side_embedding
        sum_embedding = F.leaky_relu(self.W3(add_embedding))
        bi_embedding = F.leaky_relu(self.W4(ori_embedding * side_embedding))
        entity_embed_after_encoder = sum_embedding + bi_embedding

        return entity_embed_after_encoder

    def ConvE(self, h_e, r_e, pos_t_e, neg_t_e):
        h_e = h_e.view(-1, 1, 24, 32)
        r_e = r_e.view(-1, 1, 24, 32)
        x = torch.cat([h_e, r_e], dim=2)
        x = F.relu(self.conv2d(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.Q(x))
        pos_score = torch.sigmoid((x * pos_t_e).sum(dim=-1))
        neg_score = torch.sigmoid(torch.matmul(x.unsqueeze(1), neg_t_e.transpose(1, 2)).squeeze(1))

        return pos_score, neg_score

    def ConvE_for_neg(self, h_e, r_e, t_e):
        h_e = h_e.view(-1, 1, 24, 32)
        r_e = r_e.view(-1, 1, 24, 32)
        x = torch.cat([h_e, r_e], dim=2)
        x = F.relu(self.conv2d(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.Q(x))
        score = torch.sigmoid((x * t_e).sum(dim=-1))

        return score

    def _generate_h_r_score(self, h, r):
        h_e = self.entity_embed(h)
        r_e = self.relation_embed(r)
        a_hr = self.W1(torch.cat([h_e, r_e], dim=-1))
        p_matrix = self.p.expand_as(a_hr)
        hr_score = F.leaky_relu((a_hr * p_matrix).sum(dim=-1), negative_slope=0.2)
        return hr_score

    def _generate_h_r_v_score(self, h, r, t):
        h_e = self.entity_embed(h)
        t_e = self.entity_embed(t)
        r_e = self.relation_embed(r)
        a_hr = self.W1(torch.cat([h_e, r_e], dim=-1))
        b_hrt = self.W2(torch.cat([a_hr, t_e], dim=-1))
        q_matrix = self.q.expand_as(b_hrt)
        hrt_score = F.leaky_relu((b_hrt * q_matrix).sum(dim=-1), negative_slope=0.2)
        return hrt_score, b_hrt

    def update_attentive_hr(self):
        for head_idx in self.train_all_relation_dict.keys():
            r_lst = self.train_all_relation_dict[head_idx]
            h_lst = [head_idx] * len(r_lst)
            h_tensor = torch.tensor(h_lst, dtype=torch.long).to(self.device)
            r_tensor = torch.tensor(r_lst, dtype=torch.long).to(self.device)
            score = self._generate_h_r_score(h_tensor, r_tensor)
            hr_score = torch.softmax(score, dim=0)
            for r, s in zip(r_lst, hr_score):
                self.hr_matrix[head_idx][r] = s

    def update_attentive_hrv(self):
        B_hrt = []
        for head_idx in range(self.n_entities):
            if head_idx in self.train_all_kg_dict:
                rt_matrix = torch.zeros((self.n_relations, self.n_entities), dtype=torch.float32).to(self.device)
                kg_data = self.train_all_kg_dict[head_idx]
                h_tensor = torch.full((len(kg_data),), head_idx, dtype=torch.int32, device=self.device)
                r_tensor = torch.tensor([r for r, t in kg_data], dtype=torch.int32, device=self.device)
                t_tensor = torch.tensor([t for r, t in kg_data], dtype=torch.int32, device=self.device)

                hrv_score, b_hrt = self._generate_h_r_v_score(h_tensor, r_tensor, t_tensor)
                for i, kg in enumerate(kg_data):
                    B_hrt.append([[head_idx] + list(kg), b_hrt[i]])

                rt_matrix.index_put_((r_tensor, t_tensor), hrv_score)
                mask = (rt_matrix != 0)
                rt_matrix_mask = rt_matrix.clone()
                rt_matrix_mask[~mask] = float('-inf')
                s_rt_matrix = torch.softmax(rt_matrix_mask, dim=1)
                s_rt_matrix = s_rt_matrix.masked_fill(~mask, 0.0)
                rt_matrix = s_rt_matrix
                hr_row = self.hr_matrix[head_idx]
                hrt_matrix = hr_row.reshape(-1, 1) * rt_matrix
                self.b_hrt[head_idx] = hrt_matrix
            else:
                self.b_hrt[head_idx] = torch.zeros((self.n_relations, self.n_entities), dtype=torch.float32).to(self.device)
        self.B_hrt = B_hrt

    def generate_fp_weight(self):
        with torch.no_grad():
            # false positive triplet
            fp_h, fp_r, fp_t = self.kg_neg_with_weight[:, 0], self.kg_neg_with_weight[:, 1], self.kg_neg_with_weight[:, 2]
            fp_h_e, fp_r_e, fp_t_e = self._get_kg_inference_for_neg(fp_h, fp_r, fp_t)
            ws_h_r_t = self.ConvE_for_neg(fp_h_e, fp_r_e, fp_t_e)
            ws_h_r_t = torch.exp(ws_h_r_t/self.temperature) / torch.sum(torch.exp(ws_h_r_t/self.temperature))
            ws_h_r_t = (ws_h_r_t - ws_h_r_t.min()) / (ws_h_r_t.max() - ws_h_r_t.min())
            self.wc_h_r_t = self.kg_neg_weight * ws_h_r_t

    def find_common_rows(self, tensor1, tensor2):
        matches = (tensor1.unsqueeze(1) == tensor2.unsqueeze(0)).all(dim=2)
        common_mask = matches.any(dim=1)
        common_indices = torch.nonzero(common_mask).flatten()
        common_rows = tensor1[common_indices]

        return common_rows, common_indices

    def forward(self, h, r, pos_t, neg_t, kg_refinement, idx):
        self.hr_matrix = self.hr_matrix.detach()
        self.b_hrt = [i.detach() for i in self.b_hrt]
        if self.B_hrt is not None:
            self.B_hrt = [[i[0], i[1].detach()] for i in self.B_hrt]
        if idx == 0:
            self.update_attentive_hr()
            self.update_attentive_hrv()

        self.entity_embed_after_encoder = self._create_bi_interaction_embed()
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_inference(h, r, pos_t, neg_t)
        pos_score, neg_score = self.ConvE(h_e, r_e, pos_t_e, neg_t_e)
        kge_loss = - torch.log(pos_score + 1e-24) - torch.log(1 - neg_score + 1e-24).sum(dim=1)
        kge_loss = kge_loss.mean()
        reg_loss = (h_e.norm(2).pow(2) + r_e.norm(2).pow(2) + pos_t_e.norm(2).pow(2) + neg_t_e.norm(2).pow(2)) / (
                    self.batch_size_kg * 2)
        if kg_refinement:
            if self.wc_h_r_t is None:
                self.generate_fp_weight()

            batch_triplet = torch.stack((h, r, pos_t), dim=0).T

            # false positive triplet
            fp_triplet = self.kg_neg_with_weight
            batch_fp_triplet, batch_fp_indices = self.find_common_rows(fp_triplet, batch_triplet)
            if batch_fp_indices.shape[0] != 0:
                b_fp_h, b_fp_r, b_fp_t = batch_fp_triplet[:, 0], batch_fp_triplet[:, 1], batch_fp_triplet[:, 2]
                b_fp_h_e, b_fp_r_e, b_fp_t_e = self._get_kg_inference_for_neg(b_fp_h, b_fp_r, b_fp_t)
                b_fp_score = self.ConvE_for_neg(b_fp_h_e, b_fp_r_e, b_fp_t_e)
                b_wc_h_r_t = self.wc_h_r_t[batch_fp_indices]
                fp_kge_loss = - b_wc_h_r_t * torch.log(1 - b_fp_score + 1e-24)
                fp_kge_loss = fp_kge_loss.mean()
            else:
                fp_kge_loss = torch.tensor([0.0]).to(self.device)

            # true negative triplet
            tn_sample_idx = torch.randperm(self.kg_neg_without_weight.shape[0])[:2]
            true_negative_kg = self.kg_neg_without_weight[tn_sample_idx]
            tn_h, tn_r, tn_t = true_negative_kg[:, 0], true_negative_kg[:, 1], true_negative_kg[:, 2]
            tn_h_e, tn_r_e, tn_t_e = self._get_kg_inference_for_neg(tn_h, tn_r, tn_t)
            tn_score = self.ConvE_for_neg(tn_h_e, tn_r_e, tn_t_e)
            tn_kge_loss = - torch.log(1 - tn_score + 1e-24)
            tn_kge_loss = tn_kge_loss.mean()

            kge_loss = kge_loss + fp_kge_loss + tn_kge_loss
            reg_loss = (h_e.norm(2).pow(2) + r_e.norm(2).pow(2) + pos_t_e.norm(2).pow(2) + neg_t_e.norm(2).pow(2)) / (self.batch_size_kg * 2)

        reg_loss = self.regs * reg_loss
        loss = kge_loss + reg_loss

        return loss

    def evaluate(self, h, r, t):
        h_e = self.entity_embed_after_encoder[h]
        r_e = self.relation_embed(r)
        t_e = self.entity_embed_after_encoder[t]
        h_e = h_e.view(-1, 1, 24, 32)
        r_e = r_e.view(-1, 1, 24, 32)
        x_e = torch.cat([h_e, r_e], dim=2)
        x_e = F.relu(self.conv2d(x_e))
        x_e = x_e.view(x_e.size(0), -1)
        x_e = F.relu(self.Q(x_e))
        score = torch.sigmoid((x_e * t_e).sum(dim=-1))

        return score
