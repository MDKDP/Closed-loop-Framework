import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
from RGHAT import rghat_training
from MCTS import construct_graph, monte_search
from time import strftime, localtime


def kg_pretrain():
    train_time = strftime('%Y-%m-%d%H-%M-%S', localtime())
    kg_refinement, kg_expansion, kg_pretrain = False, False, True
    rghat_training(save_flag=True, kg_refinement=kg_refinement, kg_expansion=kg_expansion, kg_pretrain=kg_pretrain, training_epoch=None, train_time=train_time, initial_rghat_path=None, seed=42)
    construct_graph(save_flag=True, kg_refinement=kg_refinement, training_epoch=None, train_time=train_time)
    monte_search(save_flag=True, kg_refinement=kg_refinement, training_epoch=None, train_time=train_time, seed=42)


def kg_expansion():
    train_time = strftime('%Y-%m-%d%H-%M-%S', localtime())
    kg_refinement, kg_expansion, kg_pretrain = False, True, False
    rghat_training(save_flag=True, kg_refinement=kg_refinement, kg_expansion=kg_expansion, kg_pretrain=kg_pretrain, training_epoch=None, train_time=train_time, initial_rghat_path=None, seed=42)
    construct_graph(save_flag=True, kg_refinement=kg_refinement, training_epoch=None, train_time=train_time)
    monte_search(save_flag=True, kg_refinement=kg_refinement, training_epoch=None, train_time=train_time, seed=42)


def kg_refinement(train_time, training_epoch, initial_rghat_path):
    kg_refinement, kg_expansion, kg_pretrain = True, False, False
    rghat_training(save_flag=True, kg_refinement=kg_refinement, kg_expansion=kg_expansion, kg_pretrain=kg_pretrain, training_epoch=training_epoch, train_time=train_time, initial_rghat_path=initial_rghat_path, seed=42)
    construct_graph(save_flag=True, kg_refinement=kg_refinement, training_epoch=training_epoch, train_time=train_time)
    monte_search(save_flag=True, kg_refinement=kg_refinement, training_epoch=training_epoch, train_time=train_time, seed=42)


if __name__ == '__main__':
    # pretrain the webmd kg
    kg_pretrain()
    # # after get the expanded kg, keep training the kg
    # kg_expansion()
    # # refine the kg
    # kg_refinement(train_time='2025-08-2415-40-40', training_epoch=0, initial_rghat_path='2025-08-2415-40-40')
