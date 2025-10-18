import pickle
import pandas as pd
import numpy as np
import torch
import math
import os
from tqdm import tqdm


def evaluation(model, kg_refinement, data_generator, device='cuda'):
    """
    Evaluate the model on the test set using HR@10 and NDCG@10 metrics.
    """
    if kg_refinement:
        neg_dict = [tuple(i) for i in np.concatenate((data_generator.kg_neg_with_weight, data_generator.kg_neg_without_weight))]

    def sample_neg_triples_for_h(h, r, num):
        neg_ts = []
        while len(neg_ts) < num:
            t = np.random.randint(low=0, high=data_generator.n_entities, size=1)[0]
            if kg_refinement:
                if (r, t) not in data_generator.test_all_kg_dict.get(h, []) and tuple([h, r, t]) not in neg_dict and t not in neg_ts:
                    neg_ts.append(t)
            else:
                if (r, t) not in data_generator.test_all_kg_dict.get(h, []) and t not in neg_ts:
                    neg_ts.append(t)
        return neg_ts

    _K = 10  # Top-K value for HR@10 and NDCG@10
    hr_lst, ndcg_lst = [], []  # Lists to store HR and NDCG scores

    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        model.update_attentive_hr()
        model.update_attentive_hrv()
        model.entity_embed_after_encoder = model._create_bi_interaction_embed()
        for head in tqdm(data_generator.test_all_kg_dict.keys(), desc="RGHAT Evaluating"):
            for triplet in data_generator.test_all_kg_dict[head]:
                relation = triplet[0]
                pos_t = triplet[1]
                tile_list = [pos_t] + sample_neg_triples_for_h(head, relation, 99)  # 1 positive + 99 negatives

                # Prepare batch tensors
                h_batch = torch.tensor([head] * len(tile_list), dtype=torch.long).to(device)
                r_batch = torch.tensor([relation] * len(tile_list), dtype=torch.long).to(device)
                t_batch = torch.tensor(tile_list, dtype=torch.long).to(device)

                scores = model.evaluate(h_batch, r_batch, t_batch)

                # Split predictions into positive and negative
                pos_score = scores[0].cpu().item()
                neg_scores = scores[1:].cpu().numpy()

                # Calculate ranking position of the positive sample
                position = (neg_scores >= pos_score).sum()

                # Compute HR@10 and NDCG@10
                hr = 1 if position < _K else 0
                ndcg = math.log(2) / math.log(position + 2) if hr else 0

                hr_lst.append(hr)
                ndcg_lst.append(ndcg)

    # Return mean metrics
    return np.mean(hr_lst), np.mean(ndcg_lst)


def test(model, optimizer, config, args, kg_refinement, kg_expansion, kg_pretrain, training_epoch, train_time, save, device='cuda'):
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        model.update_attentive_hr()
        model.update_attentive_hrv()
        model.entity_embed_after_encoder = model._create_bi_interaction_embed()
        # Get entity and relation embeddings
        all_h_list = config['train_all_h_list']
        all_r_list = config['train_all_r_list']
        all_t_list = config['train_all_t_list']
        all_v_list = []
        for i in range(0, len(all_h_list), args.batch_size_kg):
            end = min(i + args.batch_size_kg, len(all_h_list))
            h_batch = torch.tensor(all_h_list[i:end], dtype=torch.int32).to(device)
            r_batch = torch.tensor(all_r_list[i:end], dtype=torch.int32).to(device)
            t_batch = torch.tensor(all_t_list[i:end], dtype=torch.int32).to(device)

            scores = model.evaluate(h_batch, r_batch, t_batch)
            all_v_list.extend(scores.cpu().numpy().tolist())

    if save:
        if kg_expansion or kg_pretrain:
            save_dir = "../../results/JDeC/initial_result/{0}".format(train_time)
            os.makedirs(save_dir, exist_ok=True)
        if kg_refinement:
            save_dir = "../../results/JDeC/training_result/{0}/training_epoch_{1}".format(train_time, training_epoch)
            os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "all_b_hrt.pkl"), 'wb') as f2:
            pickle.dump(model.b_hrt, f2)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(save_dir, "rghat_checkpoint.pth"))

        result_data = pd.DataFrame({'head': all_h_list, 'relation': all_r_list, 'tile': all_t_list, 'value': all_v_list})
        result_data.to_csv(os.path.join(save_dir, 'rghat_triplet_score.csv'), index=False)
