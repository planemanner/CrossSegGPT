import argparse
from trainer import train_cross_seggpt

def main(args):
    if args.train:
        train_cross_seggpt(args)

    if args.evaluation:
        """
        Quant & Qual Evals.
        """
        pass
    
    if args.visualization:
        """
        Visualization Only
        """
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # all arguments are action triggers
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--devices', nargs='+', type=int)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--mlflow_db_uri', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
