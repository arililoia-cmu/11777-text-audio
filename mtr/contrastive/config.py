import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MSD Training')

    parser.add_argument('--data_path', type=str, default="../../dataset.nosync/")

    parser.add_argument('--framework', type=str, default="contrastive") # or transcription
    parser.add_argument("--text_backbone", default="bert-base-uncased", type=str)
    parser.add_argument('--arch', default='transformer')
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--min_lr', default=1e-9, type=float)
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:12312', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    
    # train detail
    parser.add_argument("--duration", default=9.91, type=int)
    parser.add_argument("--sr", default=16000, type=int)
    parser.add_argument("--num_chunks", default=3, type=int)
    parser.add_argument("--mel_dim", default=128, type=int)
    parser.add_argument("--n_fft", default=1024, type=int)
    parser.add_argument("--win_length", default=1024, type=int)
    parser.add_argument("--frontend", default="cnn", type=str)
    parser.add_argument("--mix_type", default="cf", type=str)
    parser.add_argument("--audio_rep", default="mel", type=str)
    parser.add_argument("--cos", default=True, type=bool)
    parser.add_argument("--attention_nlayers", default=4, type=int)
    parser.add_argument("--attention_ndim", default=256, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--mlp_dim", default=128, type=int)
    parser.add_argument("--text_type", default="bert", type=str)
    parser.add_argument("--text_rep", default="caption", type=str)

    return parser