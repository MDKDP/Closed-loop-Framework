import torch
from .loader_rghat import RGHAT_loader
from tqdm import tqdm
import numpy as np
from time import time, strftime, localtime
from .RGHAT import RGHAT
from .batch_test import evaluation, test
from .parser_rghat import parse_args
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def rghat_training(save_flag, kg_refinement, kg_expansion, kg_pretrain, training_epoch, train_time, initial_rghat_path, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get parameter
    args_rghat = parse_args()
    data_generator = RGHAT_loader(args_rghat, train_time, training_epoch, kg_refinement, kg_expansion, kg_pretrain)

    # Add logging
    log_dir = "../../logs/log_g/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "log_{0}".format(train_time)),
        level=logging.INFO
    )
    logging.info("KG training")
    print(args_rghat)
    logging.info(args_rghat)

    # Load data
    config = {
        'kg_refinement': data_generator.kg_refinement,
        'kg_expansion': data_generator.kg_expansion,
        'kg_pretrain': data_generator.kg_pretrain,
        'n_relations': data_generator.n_relations,
        'n_entities': data_generator.n_entities,
        'train_adj_list': data_generator.train_adj_list,
        'train_adj_r_list': data_generator.train_adj_r_list,
        'train_all_kg_dict': data_generator.train_all_kg_dict,
        'train_all_relation_dict': data_generator.train_all_relation_dict,
        'test_adj_list': data_generator.test_adj_list,
        'test_adj_r_list': data_generator.test_adj_r_list,
        'test_all_kg_dict': data_generator.test_all_kg_dict,
        'test_all_relation_dict': data_generator.test_all_relation_dict,
        'hr_norm': data_generator.hr_norm,
        'hrt_norm': data_generator.hrt_norm,
        'train_all_h_list': data_generator.train_all_h_list,
        'train_all_r_list': data_generator.train_all_r_list,
        'train_all_t_list': data_generator.train_all_t_list,
        'test_all_h_list': data_generator.test_all_h_list,
        'test_all_r_list': data_generator.test_all_r_list,
        'test_all_t_list': data_generator.test_all_t_list,
        'pretrain_emb': np.load('../../datasets/data_g/emb_matrix.pkl', allow_pickle=True)
    }
    if kg_refinement:
        config['kg_neg_with_weight'] = data_generator.kg_neg_with_weight
        config['kg_neg_weight'] = data_generator.kg_neg_weight
        config['kg_neg_without_weight'] = data_generator.kg_neg_without_weight

    # Initial the rghat
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model = RGHAT(data_config=config, args=args_rghat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_rghat.lr)

    if kg_pretrain:
        # pretrain the webmd kg, no initial rghat parameter
        pass
    if kg_refinement:
        # refine the kg, load current rghat state dict
        checkpoint_path = '../../results/JDeC/initial_result/{0}/rghat_checkpoint.pth'.format(initial_rghat_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if kg_expansion:
        # after get the expanded kg, keep training the kg, load current rghat state dict
        checkpoint_path = '../../results/JDeC/last_round_checkpoint/rghat_checkpoint.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        del checkpoint['model_state_dict']['entity_embed.weight']
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Train and Test
    max_hr = 0.0
    stopping_step = 0

    for epoch in range(args_rghat.epoch):
        t1 = time()
        model.train()
        total_loss = 0.0
        n_batch = int(np.ceil(len(data_generator.train_all_h_list) / args_rghat.batch_size_kg))
        for idx in tqdm(range(n_batch), desc="Processing Epoch {0}".format(epoch)):
            a_batch_data = data_generator.generate_train_A_batch()
            h, r, pos_t, neg_t = (a_batch_data['heads'].to(device), a_batch_data['relations'].to(device), a_batch_data['pos_tails'].to(device), a_batch_data['neg_tails'].to(device))
            optimizer.zero_grad()
            loss = model(h, r, pos_t, neg_t, kg_refinement, idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            loss_info = f"Epoch {epoch} [{time() - t1:.1f}s]: train loss={total_loss:.5f}"
            print(loss_info)
            logging.info(loss_info)

        hr, ndcg = evaluation(model, kg_refinement, data_generator)
        evaluation_str = f"Epoch {epoch}: hr@10={hr:.5f}, ndcg@10={ndcg:.5f}"
        print(evaluation_str)
        logging.info(evaluation_str)

        if hr > max_hr:
            max_hr = hr
            test(model, optimizer, config, args_rghat, kg_refinement, kg_expansion, kg_pretrain, training_epoch, train_time, save=save_flag)
            stopping_step = 0
        else:
            stopping_step += 1
        if stopping_step >= 5:
            break


if __name__ == '__main__':
    train_time = strftime('%Y-%m-%d%H-%M-%S', localtime())
    save_flag, kg_refinement, kg_expansion, kg_pretrain = True, False, False, True
    rghat_training(save_flag, kg_refinement, kg_expansion, kg_pretrain, training_epoch=None, train_time=train_time, initial_rghat_path=None, seed=42)
