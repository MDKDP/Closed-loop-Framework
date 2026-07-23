import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import os
from datasets import Dataset
import math
import multiprocessing
import pickle
from peft import PeftModel

# =====================================================
# 层次汤普森采样 (Hierarchical Thompson Sampling) 全局状态
# =====================================================
CATEGORY_POSTERIORS = {}          # 类别 -> (mu, sigma^2)
GLOBAL_THETA_MU = None            # 全局线性参数后验均值 (d,1)
GLOBAL_THETA_SIGMA = None         # 全局线性参数后验协方差 (d,d)
OBS_NOISE_VAR = 0.1               # 观测噪声方差
PRIOR_MU = 0.0                    # 类别先验均值
PRIOR_SIGMA2 = 1.0                # 类别先验方差
ALPHA = 0.5                       # 连通性权重
BETA = 0.5                        # 新颖性权重
EPSILON = 1e-5
TOP_K_ENTITIES = 5                # 每轮选中的实体数

def init_hierarchical_state(embed_dim):
    """初始化层次采样的全局后验（需在训练前调用一次）"""
    global GLOBAL_THETA_MU, GLOBAL_THETA_SIGMA
    GLOBAL_THETA_MU = np.zeros((embed_dim, 1))
    GLOBAL_THETA_SIGMA = np.eye(embed_dim)

def get_category_posterior(cat_name):
    if cat_name not in CATEGORY_POSTERIORS:
        CATEGORY_POSTERIORS[cat_name] = (PRIOR_MU, PRIOR_SIGMA2)
    return CATEGORY_POSTERIORS[cat_name]

def update_category_posterior(cat_name, reward_avg, obs_var=OBS_NOISE_VAR):
    mu_prior, sigma2_prior = get_category_posterior(cat_name)
    sigma2_post = 1.0 / (1.0/sigma2_prior + 1.0/obs_var)
    mu_post = sigma2_post * (mu_prior/sigma2_prior + reward_avg/obs_var)
    CATEGORY_POSTERIORS[cat_name] = (mu_post, sigma2_post)

def update_global_theta_posterior(X_selected, Y_selected, obs_var=OBS_NOISE_VAR):
    global GLOBAL_THETA_MU, GLOBAL_THETA_SIGMA
    X = np.array(X_selected)
    y = np.array(Y_selected).reshape(-1, 1)
    Sigma_inv = np.linalg.inv(GLOBAL_THETA_SIGMA) + (X.T @ X) / obs_var
    new_Sigma = np.linalg.inv(Sigma_inv)
    new_mu = new_Sigma @ (np.linalg.inv(GLOBAL_THETA_SIGMA) @ GLOBAL_THETA_MU + X.T @ y / obs_var)
    GLOBAL_THETA_MU = new_mu
    GLOBAL_THETA_SIGMA = new_Sigma

def thompson_sample_category(categories):
    samples = []
    for cat in categories:
        mu, sigma2 = get_category_posterior(cat)
        samples.append(np.random.normal(mu, np.sqrt(sigma2)))
    return samples

def thompson_sample_theta():
    return np.random.multivariate_normal(GLOBAL_THETA_MU.flatten(), GLOBAL_THETA_SIGMA).reshape(-1, 1)

def compute_entity_reward(entity_embed, cooccur_count, max_cooccur, ontology_embedding):
    """计算实体的基础奖励：α·连通性 + β·新颖性"""
    link_num = cooccur_count / (max_cooccur + EPSILON)
    sims = cosine_similarity(entity_embed.reshape(1, -1), ontology_embedding)[0]
    max_sim = np.max(sims)
    novelty = (1.0 - max_sim + 1.0) / 2.0
    return ALPHA * link_num + BETA * novelty


def ner_for_graph_update(data_generator, NER_results, train_time, epoch, device):
    # ---------- 1. 数据提取----------
    pos_data, neg_data = [], []
    pos_sentence_id, neg_sentence_id = [], []
    pos_token_id, neg_token_id = [], []
    pos_pred, neg_pred = [], []

    current_post_idx = 0
    for i, d in enumerate(data_generator.train_data):
        post_length = len(d['token'])
        if d['label'] == 1:
            pos_data.append(d['token'])
            b_input_data = d['Dataset']
            b_input_data = {k: torch.tensor(v).to(device) for k, v in b_input_data.items()}
            pos_pred_score = NER_results[current_post_idx:current_post_idx+post_length]
            p_pred = np.argmax(pos_pred_score, axis=-1)
            mask = np.array(d['Dataset']['labels']) != -100
            p_preds = [p_pred[i][mask[i]].tolist() for i in range(p_pred.shape[0])]
            pos_pred.append(p_preds)
            useful_token_id = b_input_data['input_ids'].cpu().numpy()
            pos_token_id.append([useful_token_id[i][mask[i]].tolist() for i in range(useful_token_id.shape[0])])
            pos_sentence_id.append(d['sentence_id'])
            torch.cuda.empty_cache()
        else:
            neg_data.append(d['token'])
            b_input_data = d['Dataset']
            b_input_data = {k: torch.tensor(v).to(device) for k, v in b_input_data.items()}
            neg_pred_score = NER_results[current_post_idx:current_post_idx+post_length]
            n_pred = np.argmax(neg_pred_score, axis=-1)
            mask = np.array(d['Dataset']['labels']) != -100
            n_preds = [n_pred[i][mask[i]].tolist() for i in range(n_pred.shape[0])]
            neg_pred.append(n_preds)
            useful_token_id = b_input_data['input_ids'].cpu().numpy()
            neg_token_id.append([useful_token_id[i][mask[i]].tolist() for i in range(useful_token_id.shape[0])])
            neg_sentence_id.append(d['sentence_id'])
            torch.cuda.empty_cache()
        current_post_idx += post_length

    def extract_depression_entities(entity_lst):
        entity_lst = [i[0] for i in entity_lst]
        final_indices = []
        i = 0
        while i < len(entity_lst):
            if entity_lst[i] == 1:
                start_idx = i
                end_idx = i + 1
                while end_idx < len(entity_lst) and entity_lst[end_idx] == 2:
                    end_idx += 1
                final_indices.append([j for j in range(start_idx, end_idx)])
                i = end_idx
            else:
                i += 1
        return final_indices

    def convert2word(name_index, pred, token, entity_name):
        converted_nameindex, converted_pred, converted_token = [], [], []
        c_nameindex, c_pred, c_token = [name_index[0]], [pred[0]], [token[0]]
        for i in range(1, len(name_index)):
            if name_index[i] == name_index[i-1]:
                c_nameindex.append(name_index[i])
                c_pred.append(pred[i])
                c_token.append(token[i])
            else:
                converted_nameindex.append(c_nameindex)
                converted_pred.append(c_pred)
                converted_token.append(c_token)
                c_nameindex = [name_index[i]]
                c_pred = [pred[i]]
                c_token = [token[i]]
        converted_nameindex.append(c_nameindex)
        converted_pred.append(c_pred)
        converted_token.append(c_token)
        entity_name = entity_name[:len(converted_nameindex)]
        for i, p in enumerate(converted_pred):
            if len(p) > 1 and (1 in p or 2 in p):
                converted_pred[i] = [1] * len(p)
        return converted_nameindex, converted_pred, converted_token, entity_name

    Pos_token, Neg_token = [], []
    Pos_token_id, Neg_token_id = [], []
    for p_index in range(len(pos_pred)):
        cur_pos_pred = pos_pred[p_index]
        cur_pos_token = pos_data[p_index]
        cur_pos_token_id = pos_token_id[p_index]
        cur_pos_sentence_id = pos_sentence_id[p_index]
        for c_i1 in range(len(cur_pos_pred)):
            if len(cur_pos_token[c_i1]) > 0:
                nameidx, pred, tokenid, entity = convert2word(cur_pos_sentence_id[c_i1][1:], cur_pos_pred[c_i1], cur_pos_token_id[c_i1], cur_pos_token[c_i1])
                depression_idx = extract_depression_entities(pred)
                if len(depression_idx) >= 2:
                    Pos_token.append([[entity[_z] for _z in z] for z in depression_idx])
                    Pos_token_id.append([[tokenid[_z] for _z in z] for z in depression_idx])
    for n_index in range(len(neg_pred)):
        cur_neg_pred = neg_pred[n_index]
        cur_neg_token = neg_data[n_index]
        cur_neg_token_id = neg_token_id[n_index]
        cur_neg_sentence_id = neg_sentence_id[n_index]
        for c_i2 in range(len(cur_neg_pred)):
            if len(cur_neg_token[c_i2]) > 0:
                nameidx, pred, tokenid, entity = convert2word(cur_neg_sentence_id[c_i2][1:], cur_neg_pred[c_i2], cur_neg_token_id[c_i2], cur_neg_token[c_i2])
                depression_idx = extract_depression_entities(pred)
                if len(depression_idx) >= 2:
                    Neg_token.append([[entity[_z] for _z in z] for z in depression_idx])
                    Neg_token_id.append([[tokenid[_z] for _z in z] for z in depression_idx])

    # 词频保存（与原代码一致）
    def word_frequency(lst):
        counter = Counter(lst)
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)
    Pos_token_freq = word_frequency([k for i in Pos_token for j in i for k in j])
    Neg_token_freq = word_frequency([k for i in Neg_token for j in i for k in j])
    save_path = '../../results/RGHAT/training_result/{0}/training_epoch_{1}'.format(train_time, epoch)
    os.makedirs(save_path, exist_ok=True)
    Pos_token_DF = pd.DataFrame(Pos_token_freq, columns=['Depression entity', 'Count'])
    Neg_token_DF = pd.DataFrame(Neg_token_freq, columns=['Depression entity', 'Count'])
    Pos_token_DF.to_csv(os.path.join(save_path, 'Pos_token_DF.csv'), index=False)
    Neg_token_DF.to_csv(os.path.join(save_path, 'Neg_token_DF.csv'), index=False)

    # ---------- 2. 构建候选实体与共现统计 ----------
    pretrain_embeddings = data_generator.text_embed_bert
    voc_id_bert = data_generator.voc_id_bert
    reverse_voc_id_bert = {v: k for k, v in voc_id_bert.items()}
    ontology_embedding = data_generator.ontology_embed_bert
    entity_list = data_generator.ontology_dict.copy()
    reverse_entity_list = data_generator.reverse_ontology_list.copy()
    existing_entities_set = set(entity_list.keys())

    def trans2bertid(entity_name):
        transed = []
        for i in entity_name:
            Id_bert = []
            for j in i:
                if len(j) == 1:
                    Id_bert.append([voc_id_bert[j[0]]] if j[0] in voc_id_bert else [-1])
                else:
                    Id_bert.append([voc_id_bert[k] if k in voc_id_bert else -1 for k in j])
            transed.append(Id_bert)
        return transed

    Pos_token_id_bert = trans2bertid(Pos_token)
    Neg_token_id_bert = trans2bertid(Neg_token)

    def get_entity_embed(token_ids):
        valid_ids = [t for t in token_ids if t != -1]
        return pretrain_embeddings[valid_ids].mean(axis=0) if valid_ids else None

    candidate_entities = {}
    def process_cooccurrence(token_id_list):
        for t_token in token_id_list:
            post_entities = []
            for t in t_token:
                if -1 not in t:
                    post_entities.append(tuple(t))
            post_entities = list(set(post_entities))
            post_new = [e for e in post_entities if e[0] not in existing_entities_set]
            post_graph = [e for e in post_entities if e[0] in existing_entities_set]
            for e_new in post_new:
                if e_new not in candidate_entities:
                    emb = get_entity_embed(e_new)
                    if emb is None: continue
                    candidate_entities[e_new] = {'embed': emb, 'cooccur_graph': Counter(), 'cooccur_new': Counter()}
                for e_kg in post_graph:
                    candidate_entities[e_new]['cooccur_graph'][e_kg[0]] += 1
                for other in post_new:
                    if other != e_new:
                        candidate_entities[e_new]['cooccur_new'][other] += 1

    process_cooccurrence(Pos_token_id_bert)
    process_cooccurrence(Neg_token_id_bert)

    if len(candidate_entities) == 0:
        return False

    max_cooccur = max((sum(d['cooccur_graph'].values()) for d in candidate_entities.values()), default=1)
    candidate_list = list(candidate_entities.keys())
    embed_matrix = np.vstack([candidate_entities[e]['embed'] for e in candidate_list])
    # 计算每个实体的基础奖励
    rewards = np.array([compute_entity_reward(candidate_entities[e]['embed'],
                                              sum(candidate_entities[e]['cooccur_graph'].values()),
                                              max_cooccur, ontology_embedding)
                        for e in candidate_list]).reshape(-1, 1)

    # ---------- 3. 实体类别映射 ----------
    entity_category_df = pd.read_csv('../../datasets/data_dep/this_round_dataset/entity_category_completed.csv')
    id_to_category = dict(zip(entity_category_df['entity_remap_id'], entity_category_df['category']))
    all_types = sorted(entity_category_df['category'].unique())

    # 为每个候选实体分配最相似的类别（使用类别的平均嵌入）
    cat_embeds = []
    valid_cats = []
    for cat in all_types:
        cat_ids = entity_category_df[entity_category_df['category'] == cat]['entity_remap_id'].values
        valid_embeds = []
        for eid in cat_ids:
            if eid < ontology_embedding.shape[0]:
                valid_embeds.append(ontology_embedding[eid])
        if valid_embeds:
            cat_embeds.append(np.mean(valid_embeds, axis=0))
            valid_cats.append(cat)
    if valid_cats:
        cat_embeds = np.vstack(cat_embeds)
        sim_to_cats = cosine_similarity(embed_matrix, cat_embeds)
        assigned_cats = [valid_cats[i] for i in np.argmax(sim_to_cats, axis=1)]
    else:
        assigned_cats = ['unknown'] * len(candidate_list)
        valid_cats = ['unknown']

    # ---------- 4. 层次汤普森采样 ----------
    ts_state_path = f'../../results/RGHAT/training_result/{train_time}/ts_state.pkl'
    if epoch == 0 or not os.path.exists(ts_state_path):
        ts_state = {
            'cat_mu': np.zeros(len(valid_cats)),
            'cat_sigma': np.ones(len(valid_cats)),
            'E_sub': []  # 存储未选中的子优实体
        }
    else:
        with open(ts_state_path, 'rb') as f:
            ts_state = pickle.load(f)

    # 高层：采样类别
    sampled = thompson_sample_category(valid_cats)
    best_cat_idx = np.argmax(sampled)
    best_cat = valid_cats[best_cat_idx]

    # 获取该类别下的候选索引
    cat_indices = [i for i, c in enumerate(assigned_cats) if c == best_cat]
    if not cat_indices:
        cat_indices = list(range(len(candidate_list)))

    # 低层：采样θ并计算预测奖励
    theta_s = thompson_sample_theta()
    pred_rewards = embed_matrix[cat_indices] @ theta_s
    K = min(TOP_K_ENTITIES, len(cat_indices))
    topk_rel_idx = np.argsort(pred_rewards.flatten())[::-1][:K]
    selected_indices = [cat_indices[i] for i in topk_rel_idx]
    selected_entities = [candidate_list[i] for i in selected_indices]

    # 未选中实体存入E_sub（用于长期探索恢复）
    for idx in cat_indices:
        if idx not in selected_indices:
            ts_state['E_sub'].append((candidate_list[idx], embed_matrix[idx], rewards[idx]))

    # 后验更新
    X_sel = embed_matrix[selected_indices]
    Y_sel = rewards[selected_indices]
    if len(X_sel) > 0:
        update_global_theta_posterior(X_sel, Y_sel)
        update_category_posterior(best_cat, np.mean(Y_sel))

    # 保存 TS 状态
    with open(ts_state_path, 'wb') as f:
        pickle.dump(ts_state, f)

    # ---------- 5. 生成正例三元组（仅使用选中实体）----------
    # 构建动态关系映射
    relation_category = {}
    rid = 0
    for i, t1 in enumerate(all_types):
        for t2 in all_types[i:]:
            relation_category[t1 + '-' + t2] = rid
            rid += 1
    for k in list(relation_category.keys()):
        parts = k.split('-')
        if parts[0] != parts[1]:
            relation_category[parts[1] + '-' + parts[0]] = relation_category[k]

    global_nodes_set = {546, 442, 402, 418, 381, 347}
    final_pos_triplets = {'triplet': [], 'triplet_name': [], 'initial_text_id': [], 'initial_text_entity': []}
    cur_new_entity = []

    for idx, e_new in enumerate(selected_entities):
        e_name = [reverse_voc_id_bert[t] for t in e_new]
        cur_new_entity.append('_'.join(e_name))
        emb = candidate_entities[e_new]['embed']
        sims = cosine_similarity(emb.reshape(1, -1), ontology_embedding)[0]
        valid_nodes = [i for i in range(len(sims)) if i not in global_nodes_set and sims[i] > 0.86]
        if not valid_nodes:
            valid_nodes = [np.argmax([sims[i] if i not in global_nodes_set else -1 for i in range(len(sims))])]
        best_node = sorted(valid_nodes, key=lambda x: sims[x], reverse=True)[:3]

        cooccur_kg = candidate_entities[e_new]['cooccur_graph']
        linked = False
        for node_id in best_node:
            if len(cooccur_kg) > 0:
                for kg_id, _ in cooccur_kg.most_common(3):
                    if kg_id in id_to_category:
                        node_cat = id_to_category.get(node_id, 'unknown')
                        kg_cat = id_to_category[kg_id]
                        rel_str = f"{node_cat}-{kg_cat}"
                        rel_id = relation_category.get(rel_str, 0)
                        final_pos_triplets['triplet'].append([node_id, rel_id, kg_id])
                        final_pos_triplets['triplet_name'].append([reverse_entity_list.get(node_id, str(node_id)),
                                                                   reverse_entity_list.get(kg_id, str(kg_id))])
                        final_pos_triplets['initial_text_id'].append([e_new, (kg_id,)])
                        final_pos_triplets['initial_text_entity'].append([e_name, [reverse_voc_id_bert.get(kg_id, '')]])
                        linked = True
                        break
            if linked:
                break

    # ---------- 6. 负例三元组----------
    Neg_e = []
    depression_embed = pretrain_embeddings[voc_id_bert['depression']]
    virus_embed = pretrain_embeddings[voc_id_bert['virus']]
    flu_embed = pretrain_embeddings[voc_id_bert['flu']]
    Neg_token_id_bert = trans2bertid(Neg_token)
    for t_token in Neg_token_id_bert:
        unique_entities = []
        for t in t_token:
            if -1 not in t:
                if t not in unique_entities:
                    if (cosine_similarity(pretrain_embeddings[t], np.expand_dims(depression_embed, axis=0))[0][0] > 0.815
                        or cosine_similarity(pretrain_embeddings[t], np.expand_dims(virus_embed, axis=0))[0][0] > 0.80
                        or cosine_similarity(pretrain_embeddings[t], np.expand_dims(flu_embed, axis=0))[0][0] > 0.80):
                        unique_entities.append(t)
        if len(unique_entities) >= 2:
            Neg_e.append(unique_entities)

    Neg_nodes = {'text_token_id': [], 'similar_graph_nodes_id': [], 'text': [], 'similar_graph_nodes': []}
    if len(Neg_e) > 0:
        neg_embeds = [pretrain_embeddings[i[0]] if len(i)==1 else pretrain_embeddings[i].mean(0) for n_e in Neg_e for i in n_e]
        neg_similarities = np.stack([cosine_similarity(ontology_embedding, np.expand_dims(i, axis=0)).squeeze(-1) for i in neg_embeds])
        idx = 0
        for n_e in Neg_e:
            n = len(n_e)
            sim = neg_similarities[idx:idx+n, :]
            idx += n
            top_3_indices = np.argpartition(-sim, 2, axis=1)[:, :3]
            top_3_sim = np.take_along_axis(sim, top_3_indices, axis=1).ravel()
            valid_rows, valid_text_id, valid_text, valid_nodes = [], [], [], []
            for idxs, (row_indices, row_sim) in enumerate(zip(top_3_indices, top_3_sim)):
                row_mask = (~np.isin(row_indices, list(global_nodes_set))) & (row_sim > 0.885)
                valid_indices = row_indices[row_mask].tolist()
                if valid_indices:
                    valid_rows.append(valid_indices)
                    valid_text_id.append(n_e[idxs])
                    valid_text.append([reverse_voc_id_bert[_i] for _i in n_e[idxs]])
                    valid_nodes.append([reverse_entity_list[_i] for _i in valid_indices])
            if len(valid_rows) >= 2:
                Neg_nodes['text_token_id'].append(valid_text_id)
                Neg_nodes['similar_graph_nodes_id'].append(valid_rows)
                Neg_nodes['text'].append(valid_text)
                Neg_nodes['similar_graph_nodes'].append(valid_nodes)

    Neg_type = []
    for ni in Neg_nodes['similar_graph_nodes_id']:
        ni_type = []
        for n_i in ni:
            n_i_type = []
            for n in n_i:
                n_type = entity_category_df.loc[entity_category_df['entity_remap_id'] == n]['category'].tolist()[0]
                n_i_type.append(n_type)
            ni_type.append(n_i_type)
        Neg_type.append(ni_type)

    final_neg_triplets = {'triplet': [], 'triplet_name': [], 'initial_text_id': [], 'initial_text_entity': []}
    for n, t in enumerate(Neg_type):
        cur_l = len(t)
        for i in range(cur_l-1):
            for j in range(i+1, cur_l):
                h_lst, t_lst = t[i], t[j]
                for h_, _h in enumerate(h_lst):
                    for t_, _t in enumerate(t_lst):
                        if _h + '-' + _t in relation_category:
                            rel = _h + '-' + _t
                        else:
                            rel = _t + '-' + _h
                        rel_id = relation_category[rel]
                        final_neg_triplets['triplet'].append([Neg_nodes['similar_graph_nodes_id'][n][i][h_], rel_id, Neg_nodes['similar_graph_nodes_id'][n][j][t_]])
                        final_neg_triplets['triplet_name'].append([Neg_nodes['similar_graph_nodes'][n][i][h_], Neg_nodes['similar_graph_nodes'][n][j][t_]])
                        final_neg_triplets['initial_text_id'].append([Neg_nodes['text_token_id'][n][i], Neg_nodes['text_token_id'][n][j]])
                        final_neg_triplets['initial_text_entity'].append([Neg_nodes['text'][n][i], Neg_nodes['text'][n][j]])

    # 保存正/负例三元组
    pd.DataFrame(final_pos_triplets).to_csv(os.path.join(save_path, 'detected_with_depression_entities.csv'), index=False)
    pd.DataFrame(final_neg_triplets).to_csv(os.path.join(save_path, 'detected_without_depression_entities.csv'), index=False)

    # 更新新实体列表
    if epoch != 0 and os.path.exists(f'../../results/RGHAT/training_result/{train_time}/new_entities.csv'):
        last_entities = pd.read_csv(f'../../results/RGHAT/training_result/{train_time}/new_entities.csv')['new_entities'].tolist()
        cur_new_entity = list(set(cur_new_entity + last_entities))
    pd.DataFrame(cur_new_entity, columns=['new_entities']).to_csv(
        f'../../results/RGHAT/training_result/{train_time}/new_entities.csv', index=False, encoding='utf-8')

    print(f"Detect new entity from post successfully. Selected {len(selected_entities)} entities using Thompson Sampling.")
    return len(final_pos_triplets['triplet']) > 0

def use_model_for_ner(dataset, model, data_generator, args, device, data_type):
    batch_size = args.batch_size_ner
    seq_length_dep = args.maxLengthDEP
    model.eval()
    with torch.no_grad():
        attention_mask = np.array(dataset['attention_mask'])
        mask = attention_mask != 0
        NER_results = []
        true_hidden_states = []
        n_batch_ner = int(np.ceil(len(dataset) / batch_size))
        for idx in tqdm(range(n_batch_ner), desc="NER Processing"):
            batch_ner_data = data_generator.generate_batch_ner(idx, dataset, data_type=data_type)
            batch_ner_data = {k: torch.tensor(v).to(device) for k, v in batch_ner_data.items()}
            logits, hidden_states = model(**batch_ner_data, NER_OR_DEP='NER')
            NER_results.append(logits.cpu().numpy())
            hidden_states = hidden_states.half().cpu().numpy()
            batch_mask = mask[idx*batch_size:min((idx+1)*batch_size, len(dataset))]
            for i in range(hidden_states.shape[0]):
                true_hidden_states.append(hidden_states[i][batch_mask[i]])
            torch.cuda.empty_cache()

        NER_results = np.concatenate(NER_results, axis=0)
        NER_prediction = np.argmax(NER_results, axis=-1)

        if data_type == 'train':
            data = data_generator.train_data
        elif data_type == 'test':
            data = data_generator.test_data
        elif data_type == 'final_test':
            data = data_generator.final_test_data
        else:
            raise ValueError("Wrong data type: ", data_type)
        input_ids = np.array([j for i in data for j in i['Dataset']['input_ids']])
        NER_predictions = [NER_prediction[i][mask[i]].tolist() for i in range(attention_mask.shape[0])]
        true_input_ids = [input_ids[i][mask[i]].tolist() for i in range(attention_mask.shape[0])]

        NER_predictions = [i[1:] for i in NER_predictions]
        true_input_ids = [i[1:] for i in true_input_ids]
        true_hidden_states = [i[1:] for i in true_hidden_states]

        attn_masks = [np.array(j) for i in data for j in i['Dataset']['attention_mask']]
        sentence_id_flat = [j for i in data for j in i['sentence_id']]
        sentence_id = []
        for sid, am in zip(sentence_id_flat, attn_masks):
            filtered = [s for s, m in zip(sid[1:], am[1:]) if m == 1]
            sentence_id.append(filtered)
        NER_predictions = convert_predictions(NER_predictions, sentence_id)

        cur_idx = 0
        for i in range(len(data)):
            cur_data = data[i]
            token_num = len(cur_data['token'])
            token_name = cur_data['token']
            ner_prediction = [NER_predictions[c] for c in range(cur_idx, cur_idx+token_num)]
            true_input_id = [true_input_ids[c] for c in range(cur_idx, cur_idx+token_num)]
            true_hidden_state = [true_hidden_states[c] for c in range(cur_idx, cur_idx+token_num)]

            ner_entity_id = []
            ner_entity_embed = []
            ner_entity_name = []
            entity_sentence_id = []
            srt_sent_id = 0
            for j in range(token_num):
                mask = np.array(ner_prediction[j]) != 0
                input_id = np.array(true_input_id[j])[mask].tolist()
                hidden_state = true_hidden_state[j][mask]
                words_id = data[i]['sentence_id'][j][1:]
                words_idx = [words_id[w] for w in np.where(mask)[0]]
                if len(words_idx) != 0:
                    _, indics = np.unique(np.array(words_idx), return_inverse=True)
                    entity_sentence_id.append(list(indics + srt_sent_id))
                    srt_sent_id += int(max(indics) + 1)
                else:
                    entity_sentence_id.append([])
                ner_entity_id.append(input_id)
                ner_entity_embed.append(hidden_state)
                ner_entity_name.append([token_name[j][m] for m in words_idx])
            cur_idx += token_num
            ner_entity_id = ner_entity_id[::-1]
            ner_entity_embed = ner_entity_embed[::-1]
            ner_entity_name = ner_entity_name[::-1]
            entity_sentence_id = entity_sentence_id[::-1]
            if ner_entity_id != []:
                ner_entity_id = np.concatenate(ner_entity_id).astype(int)[:seq_length_dep]
                ner_entity_embed = np.concatenate(ner_entity_embed)[:seq_length_dep]
                ner_entity_name = [n for name in ner_entity_name for n in name][:seq_length_dep]
                entity_sentence_id = [int(s) for sid in entity_sentence_id for s in sid][:seq_length_dep]
            else:
                ner_entity_id = [true_input_id[0][0]]
                ner_entity_embed = true_hidden_state[0][0].reshape(1, -1)
                ner_entity_name = [token_name[0][0]]
                entity_sentence_id = [data[i]['sentence_id'][0][1]]
            data[i]['ner_entity_id'] = ner_entity_id
            data[i]['ner_entity_embed'] = ner_entity_embed
            data[i]['ner_entity_name'] = ner_entity_name
            data[i]['entity_sentence_id'] = entity_sentence_id

        if data_type == 'train':
            return NER_results
        elif data_type == 'test':
            return NER_results
        elif data_type == 'final_test':
            return NER_results


def convert_predictions(pred_list, id_list):
    result = []
    for preds, ids in zip(pred_list, id_list):
        word_groups = {}
        for idx, word_id in enumerate(ids):
            if word_id not in word_groups:
                word_groups[word_id] = []
            word_groups[word_id].append(idx)
        new_preds = preds.copy()
        for word_id, indices in word_groups.items():
            sub_preds = [preds[i] for i in indices]
            if 2 in sub_preds:
                for i in indices:
                    new_preds[i] = 2
            elif 1 in sub_preds:
                for i in indices:
                    new_preds[i] = 1
        result.append(new_preds)
    return result


def evaluate_ner(tokenizer, model, batch_size, data_generator, device, data_type):
    dataset = data_tokenizer(data_generator, tokenizer, data_type=data_type)
    model.eval()
    with torch.no_grad():
        NER_results = []
        n_batch_ner = int(np.ceil(len(dataset) / batch_size))
        for idx in tqdm(range(n_batch_ner), desc="NER Evaluating"):
            batch_ner_data = data_generator.generate_batch_ner(idx, dataset, data_type='test')
            batch_ner_data = {k: torch.tensor(v).to(device) for k, v in batch_ner_data.items()}
            logits, _ = model(**batch_ner_data, NER_OR_DEP='NER')
            NER_results.append(logits)

        NER_results = torch.concatenate(NER_results, dim=0)
        NER_prediction = torch.argmax(NER_results, dim=-1).cpu().numpy()
        labels = np.array(dataset['labels'])

        NER_labels = []
        NER_predictions = []
        for i in range(len(labels)):
            mask = labels[i] != -100
            true_label = (labels[i][mask] - 1).tolist()
            true_prediction = NER_prediction[i][mask].tolist()
            NER_labels.append(true_label)
            NER_predictions.append(true_prediction)

        labels = [j for i in NER_labels for j in i]
        labels = [1 if i == 2 else i for i in labels]
        pred = [j for i in NER_predictions for j in i]
        pred = [1 if i == 2 else i for i in pred]

        f1 = f1_score(labels, pred, pos_label=1)
        precision = precision_score(labels, pred, pos_label=1)
        recall = recall_score(labels, pred, pos_label=1)
        return [f1, precision, recall]


def save_explainable_result(explainable_record, reverse_ontology_list, train_time, epoch):
    explainable_record['top_sim_ontology_name'] = []
    for batch_record in explainable_record['top_sim_ontology_id']:
        batch_ontology_name_lst = []
        for record in batch_record:
            ontology_name_lst = []
            for ontology_ids in record:
                ontology_name_lst.append([reverse_ontology_list[_o] for _o in ontology_ids])
            batch_ontology_name_lst.append(ontology_name_lst)
        explainable_record['top_sim_ontology_name'].append(batch_ontology_name_lst)

    for k in explainable_record.keys():
        explainable_record[k] = [j for i in explainable_record[k] for j in i]
    explainable_record['text'] = [
        [x for x in t if x not in ['PAD', 'SRT']]
        for t in explainable_record['text']
    ]
    for k in ['top_sim_ontology_id', 'ontology_weight', 'entity_weight', 'top_sim_ontology_name']:
        explainable_record[k] = [
            explainable_record[k][i][-len(t):]
            for i, t in enumerate(explainable_record['text'])
        ]
    explainable_result_path = '../../results/RGHAT/training_result/{0}/training_epoch_{1}'.format(train_time, epoch)
    os.makedirs(explainable_result_path, exist_ok=True)
    pd.DataFrame(explainable_record).to_csv(os.path.join(explainable_result_path, 'test_explainable_record.csv'), index=False)


def save_final_explainable_result(explainable_record, reverse_ontology_list, train_time, epoch):
    explainable_record['top_sim_ontology_name'] = []
    for batch_record in explainable_record['top_sim_ontology_id']:
        batch_ontology_name_lst = []
        for record in batch_record:
            ontology_name_lst = []
            for ontology_ids in record:
                ontology_name_lst.append([reverse_ontology_list[_o] for _o in ontology_ids])
            batch_ontology_name_lst.append(ontology_name_lst)
        explainable_record['top_sim_ontology_name'].append(batch_ontology_name_lst)

    for k in explainable_record.keys():
        explainable_record[k] = [j for i in explainable_record[k] for j in i]
    explainable_record['text'] = [
        [x for x in t if x not in ['PAD', 'SRT']]
        for t in explainable_record['text']
    ]
    for k in ['top_sim_ontology_id', 'ontology_weight', 'entity_weight', 'top_sim_ontology_name']:
        explainable_record[k] = [
            explainable_record[k][i][-len(t):]
            for i, t in enumerate(explainable_record['text'])
        ]
    explainable_result_path = '../../results/RGHAT/training_result/{0}/training_epoch_{1}'.format(train_time, epoch)
    os.makedirs(explainable_result_path, exist_ok=True)
    pd.DataFrame(explainable_record).to_csv(os.path.join(explainable_result_path, 'final_test_explainable_record.csv'), index=False)


def update_JDeC(model, epoch, train_time, ontology_embed_bert):
    pkl_path = '../../results/RGHAT/training_result/{0}/training_epoch_{1}/record_max_pro_sqrt.pkl'.format(train_time, epoch)
    if os.path.exists(pkl_path):
        new_ontology_weight = np.load(pkl_path, allow_pickle=True)
        new_ontology_weight = np.array(new_ontology_weight).reshape([1, -1])
        new_ontology_weight = torch.tensor(new_ontology_weight, dtype=torch.float32).to(model.device)
        model.ontology_weight = new_ontology_weight
    else:
        print(f"Warning: {pkl_path} not found. Keep ontology_weight unchanged.")
    graph_pth = '../../results/JDeC/training_result/{0}/training_epoch_{1}/rghat_checkpoint.pth'.format(train_time, epoch)
    if os.path.exists(graph_pth):
        checkpoint = torch.load(graph_pth, map_location='cpu')
        graph_embed = checkpoint['model_state_dict']['entity_embed.weight']
        depression_embed = torch.from_numpy(ontology_embed_bert[-1, :]).to(graph_embed.dtype)
        graph_embed = torch.cat([graph_embed.to(depression_embed.device), depression_embed.unsqueeze(0)], dim=0)
        model.graph_embed = graph_embed
    else:
        print(f"Warning: {graph_pth} not found. Keep graph_embed unchanged.")
    return model


def data_tokenizer(data_generator, tokenizer, data_type):
    tokenizer.padding_side = 'right'
    def process_func_llama_for_NER(example, tokenizer, maxLengthNER):
        labels = example["label"]
        words = example["text"]
        word_ids = [-100]
        tokenized_tokens = []
        for word_idx, word in enumerate(words):
            tokens = tokenizer.tokenize(word)
            tokenized_tokens.extend(tokens)
            word_ids.extend([word_idx] * len(tokens))

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            max_length=maxLengthNER,
            padding='max_length',
            truncation=True,
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        aligned_labels = [-100] * maxLengthNER
        for token_idx, word_idx in enumerate(encoding.word_ids()):
            if word_idx is None:
                aligned_labels[token_idx] = -100
            else:
                aligned_labels[token_idx] = labels[word_idx]

        if input_ids[0] != tokenizer.bos_token_id:
            input_ids = [tokenizer.bos_token_id] + input_ids[:-1]
            attention_mask = [1] + attention_mask[:-1]
            aligned_labels = [-100] + aligned_labels[:-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels,
            "sentence_id": word_ids[:maxLengthNER],
        }

    if data_type == 'train':
        ner_dataset = Dataset.from_dict({
            'text': data_generator.train_tokenLst,
            'label': data_generator.train_tagLst
        })
        num_proc = max(1, multiprocessing.cpu_count() // 20)
    elif data_type == 'test':
        ner_dataset = Dataset.from_dict({
            'text': data_generator.test_tokenLst,
            'label': data_generator.test_tagLst
        })
        num_proc = 1
    elif data_type == 'final_test':
        ner_dataset = Dataset.from_dict({
            'text': data_generator.final_test_tokenLst,
            'label': data_generator.final_test_tagLst
        })
        num_proc = 1
    else:
        raise ValueError('Invalid data_type {}'.format(data_type))

    ner_dataset = ner_dataset.map(
        process_func_llama_for_NER,
        fn_kwargs={'tokenizer': tokenizer, 'maxLengthNER': data_generator.maxLengthNER},
        remove_columns=['text', 'label'],
        num_proc=num_proc
    )
    sentence_id = ner_dataset['sentence_id']
    ner_dataset = ner_dataset.remove_columns('sentence_id')
    if data_type == 'train':
        idx = 0
        for i in range(len(data_generator.train_data)):
            cur_count = len(data_generator.train_data[i]['token'])
            data_generator.train_data[i]['Dataset'] = ner_dataset[idx:idx + cur_count]
            data_generator.train_data[i]['sentence_id'] = sentence_id[idx:idx + cur_count]
            idx += cur_count
    elif data_type == 'test':
        idx = 0
        for i in range(len(data_generator.test_data)):
            cur_count = len(data_generator.test_data[i]['token'])
            data_generator.test_data[i]['Dataset'] = ner_dataset[idx:idx + cur_count]
            data_generator.test_data[i]['sentence_id'] = sentence_id[idx:idx + cur_count]
            idx += cur_count
    elif data_type == 'final_test':
        idx = 0
        for i in range(len(data_generator.final_test_data)):
            cur_count = len(data_generator.final_test_data[i]['token'])
            data_generator.final_test_data[i]['Dataset'] = ner_dataset[idx:idx + cur_count]
            data_generator.final_test_data[i]['sentence_id'] = sentence_id[idx:idx + cur_count]
            idx += cur_count
    return ner_dataset


def set_head_requires_grad(model):
    for name, parameter in model.model.NERHead.named_parameters():
        parameter.requires_grad = True
    for name, parameter in model.model.DEPHead.named_parameters():
        parameter.requires_grad = True
    for name, parameter in model.model.EmbedFusion.named_parameters():
        parameter.requires_grad = True
    return model


def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


def step_learning_rate(current_round, initial_lr, min_lr):
    decay_steps = current_round // 4
    lr = initial_lr * (0.2 ** decay_steps)
    return max(lr, min_lr)


def save_checkpoint(model, optimizer_ner, optimizer_dep, epoch, best_f1, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    model.save_pretrained(checkpoint_dir)
    torch.save({
        'epoch': epoch,
        'optimizer_ner_state_dict': optimizer_ner.state_dict(),
        'optimizer_dep_state_dict': optimizer_dep.state_dict(),
        'best_f1': best_f1,
        'NERHead_state_dict': model.NERHead.state_dict(),
        'DEPHead_state_dict': model.DEPHead.state_dict(),
        'EmbedFusion_state_dict': model.EmbedFusion.state_dict(),
    }, checkpoint_path)
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')])
    for old_checkpoint in checkpoints[:-2]:
        os.remove(os.path.join(checkpoint_dir, old_checkpoint))


def load_checkpoint(model, optimizer_ner, optimizer_dep, checkpoint_path, device):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.NERHead.load_state_dict(checkpoint['NERHead_state_dict'])
    model.DEPHead.load_state_dict(checkpoint['DEPHead_state_dict'])
    optimizer_ner.load_state_dict(checkpoint['optimizer_ner_state_dict'])
    optimizer_dep.load_state_dict(checkpoint['optimizer_dep_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['best_f1']
    return model, optimizer_ner, optimizer_dep, start_epoch, best_f1
