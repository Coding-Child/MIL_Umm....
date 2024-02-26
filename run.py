import argparse
from run_utils.runfile import main
import warnings
warnings.filterwarnings("ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters for the model.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('-w', '--loss_weight', type=float, default=0.01, help='Loss weight based on instance loss ratio.')
    parser.add_argument('-s', '--scheduler', type=str, default=None, help='Using Scheduler or not.')
    parser.add_argument('-opt', '--optimizer', type=str, default='Adam', help='Optimizer type.')
    parser.add_argument('-g', '--gamma', type=float, default=0.8, help='Gamma for StepLR.')
    parser.add_argument('-ws', '--warmup_steps', type=int, default=4000, help='Warmup steps for NoamLR.')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout prob.')
    parser.add_argument('-m', '--d_model', type=int, default=512, help='dimension of model.')
    parser.add_argument('-n', '--num_heads', type=int, default=8, help='Number of Attn Layers.')
    parser.add_argument('-l', '--num_layers', type=int, default=6, help='Number of Transformer Layers.')
    parser.add_argument('-f', '--num_fc', type=int, default=2, help='Number of Fully Connected Layers.')
    parser.add_argument('-c', '--num_classes', type=int, default=2, help='Number of Data\'s class.')
    parser.add_argument('-e', '--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('-p', '--num_patience', type=int, default=10, help='Number of patience for early stopping.')
    parser.add_argument('-ns', '--num_splits', type=int, default=5, help='Number of splits for K-Fold.')
    parser.add_argument('-pt', '--pretrained', type=bool, default=True, help='Using Pretrained model or not.')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('-i', '--num_instance', type=int, default=32, help='Training Instance size.')
    parser.add_argument('-r', '--csv_root_dir', type=str, default='data_info.csv', help='Root directory of train dataset.')

    args = parser.parse_args()
    main(args)