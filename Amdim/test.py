import os
import argparse

import torch

import mixed_precision
from stats import AverageMeterSet
from datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from model import Model
from checkpoint import Checkpointer
from utils import test_model

parser = argparse.ArgumentParser(description='Infomax Representations - Testing Script')
# parameters for general training stuff
parser.add_argument('--checkpoint_path', type=str,
                    help='path from which to load checkpoint')
parser.add_argument('--dataset', type=str, default='STL10')
parser.add_argument('--n_classes', type=int, default=8,
                   help="numbers of classes in the dataset")
parser.add_argument('--batch_size', type=int, default=200,
                    help='input batch size (default: 200)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enables automatic mixed precision')
parser.add_argument('--input_dir', type=str, default='/mnt/imagenet',
                    help="Input directory for the dataset. Not needed For C10,"
                    " C100 or STL10 as the data will be automatically downloaded.")
parser.add_argument('--run_name', type=str, default='default_run',
                    help='name to use for the tensorbaord summary for this run')

# parameters for train scene and test scene
parser.add_argument('--train_scene', type=str, required=True,
                    help='Scene folder used for training')
parser.add_argument('--test_scene', type=str, required=True,
                    help='Scene folder used for testing')
parser.add_argument('-sn', '--suffix_name', type=str, default=None)

args = parser.parse_args()


def test(model, test_loader, device, stats):
    cls_report_mlp, cls_report_lin = test_model(model, test_loader, device, stats, print_classification_report=True, suffix_name=args.suffix_name)
    return cls_report_mlp, cls_report_lin


def main():

    # enable mixed-precision computation if desired
    if args.amp:
        mixed_precision.enable_mixed_precision()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get the dataset
    dataset = get_dataset(args.dataset)

    _, test_loader, _ = build_dataset(dataset=dataset,
                            batch_size=args.batch_size,
                            input_dir=args.input_dir,
                            n_classes=args.n_classes,
                            train_scene=args.train_scene,
                            test_scene=args.test_scene)

    torch_device = torch.device('cuda')
    checkpointer = Checkpointer()
   
    model = checkpointer.restore_model_from_checkpoint(args.checkpoint_path)
    model = model.to(torch_device)
    model, _ = mixed_precision.initialize(model, None)

    test_stats = AverageMeterSet()
    cls_report_mlp, cls_report_lin = test(model, test_loader, torch_device, test_stats)
    stat_str = test_stats.pretty_string(ignore=model.tasks)
    print(stat_str)
    print('[INFO] In mlp, for each class: \n', cls_report_mlp)
    print('[INFO] In lin, for each class: \n', cls_report_lin)



if __name__ == "__main__":
    print(args)
    main()
