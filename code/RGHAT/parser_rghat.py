import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run RGHAT.")
    parser.add_argument('--epoch', type=int, default=100, help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=768, help='Embedding size.')
    parser.add_argument('--kge_size', type=int, default=768, help='KG Embedding size.')
    parser.add_argument('--batch_size_kg', type=int, default=128, help='KG batch size.')
    parser.add_argument('--regs', type=float, default=1e-5, help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature coefficient.')

    return parser.parse_args()

