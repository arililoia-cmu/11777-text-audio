import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from transformers import AutoModel, AutoTokenizer, set_seed
from tqdm import tqdm, trange
import random
import builtins
import shutil
import math
import wandb

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from dataset import ECALS_Dataset
from config import get_parser, CLUSTERS
from model import ContrastiveModel
from mtr.modules.audio_rep import TFRep
from mtr.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mtr.modules.encoder import MusicTransformer
from mtr.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams

def main():
    parser = get_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        nprocess = torch.cuda.device_count()
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    #     nprocess = 1
    else:
        device = "cpu"
        nprocess = 1
    args.device = device
    print(f"Device: {device}, Number of processes: {nprocess}")

    if args.log:
        wandb.init(project="disentangle", group="disentangle" if args.disentangle else "baseline")
        wandb.config.update(args)
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("cluster/*", step_metric="epoch")
    

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.multiprocessing_distributed:
        args.world_size = nprocess * args.world_size
        mp.spawn(main_worker, nprocs=nprocess, args=(nprocess, args))
    else:
        main_worker(nprocess, args)

def main_worker(ngpus_per_node, args):
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.device == 'cuda' and args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    '''Audio Encoder'''
    audio_preprocessr = TFRep(
                sample_rate= args.sr,
                f_min=0,
                f_max= int(args.sr / 2),
                n_fft = args.n_fft,
                win_length = args.win_length,
                hop_length = int(0.01 * args.sr),
                n_mels = args.mel_dim
    )
    frontend = ResFrontEnd(
        input_size=(args.mel_dim, int(100 * args.duration) + 1), # 128 * 992
        conv_ndim=128, 
        attention_ndim=args.attention_ndim,
        mix_type= args.mix_type
    )
    audio_encoder = MusicTransformer(
        audio_representation=audio_preprocessr,
        frontend = frontend,
        audio_rep = args.audio_rep,
        attention_nlayers= args.attention_nlayers,
        attention_ndim= args.attention_ndim,
        dropout = args.dropout
    )

    '''Text Encoder'''
    if args.text_type == "bert":
        text_encoder = AutoModel.from_pretrained(args.text_backbone)
        tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
        args.text_dim = 768
    elif args.text_type == "glove":
        text_encoder = nn.Identity()
        tokenizer = torch.load(os.path.join(args.data_dir, "msd", "glove_tag_embs.pt"))
        args.text_dim = 300
    # freeze text encoder
    if args.freeze:
        for param in text_encoder.parameters():
            param.requires_grad = False
    
    args.audio_dim = args.attention_ndim
    model = ContrastiveModel(
        audio_encoder= audio_encoder,
        text_encoder= text_encoder,
        text_type = args.text_type,
        audio_dim= args.audio_dim,
        text_dim= args.text_dim,
        mlp_dim= args.mlp_dim,
        temperature = args.temperature,
        disentangle = args.disentangle,
        n_proj = args.n_proj
    )
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M')
    print(f'Audio encoder: {sum(p.numel() for p in model.audio_encoder.parameters()) / 1e6:.0f}M')
    print(f'Text encoder: {sum(p.numel() for p in model.text_encoder.parameters()) / 1e6:.0f}M')

    train_dataset = ECALS_Dataset(
        data_path = args.data_path,
        split = "TRAIN",
        sr = args.sr,
        duration = args.duration,
        num_chunks = args.num_chunks,
        text_preprocessor= tokenizer,
        text_type=args.text_type,
        text_rep = args.text_rep,
        disentangle = args.disentangle,
        subset = args.subset
    )
    val_dataset = ECALS_Dataset(
        data_path = args.data_path,
        split = "VALID",
        sr = args.sr,
        duration = args.duration,
        num_chunks = args.num_chunks,
        text_preprocessor= tokenizer,
        text_type=args.text_type,
        text_rep = args.text_rep,
        disentangle = args.disentangle,
        subset = args.subset
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), collate_fn=train_dataset.batch_processor,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None), collate_fn=train_dataset.batch_processor,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    if args.distributed:
        model = model.to(args.device)
        # args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = model.to(args.device)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.1)
    earlystopping_callback = EarlyStopping()
    cudnn.benchmark = True

    save_dir = args.save_path
    # logger = Logger(save_dir)
    logger = None
    if args.name:
        model_name = args.name
    else:
        model_name = 'disentangle' if args.disentangle else 'base'
    save_hparams(args, save_dir, model_name)
    
    best_val_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        
        # train for one epoch
        train(train_loader, model, optimizer, epoch, logger, args)
        val_loss, audio_accs, text_accs = validate(val_loader, model, epoch, args)
        
        if args.log:
            # logger.log_val_loss(val_loss, epoch)
            # logger.log_audio_acc(audio_accs, epoch)
            # logger.log_text_acc(text_accs, epoch)
            log_items = {"epoch": epoch, "val/loss": val_loss.item()}
            if audio_accs.ndim == 0:
                log_items["val/audio_acc"] = audio_accs.item()
                log_items["val/text_acc"] = text_accs.item()
            else:
                for i, cluster in enumerate(CLUSTERS):
                    log_items[f"cluster/audio_acc_{cluster}"] = audio_accs[i].item()
                    log_items[f"cluster/text_acc_{cluster}"] = text_accs[i].item()
                log_items[f"val/audio_acc"] = audio_accs[-2].item()
                log_items[f"val/text_acc"] = text_accs[-2].item()
                log_items[f"val/weighted_audio_acc"] = audio_accs[-1].item()
                log_items[f"val/weighted_text_acc"] = text_accs[-1].item()
            wandb.log(log_items)
        
        # save model
        if val_loss < best_val_loss:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/{model_name}_best.pth')
            best_val_loss = val_loss
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/{model_name}_last.pth')
        if args.epochs > 50 and epoch % 50 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/{model_name}_{epoch}.pth')

        # earlystopping_callback(val_loss, best_val_loss)
        # if earlystopping_callback.early_stop:
        #     print("We are at epoch:", epoch)
        #     break

def train(train_loader, model, optimizer, epoch, logger, args):
    # train_losses = AverageMeter('Train Loss', ':.4e')
    # progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()
    for data_iter_step, batch in enumerate(tqdm(train_loader)):
        lr = adjust_learning_rate(optimizer, (data_iter_step + 1) / iters_per_epoch + epoch, args)
        audio = batch['audio']
        text = batch['text']
        text_mask = batch['text_mask']
        cluster_mask = batch['cluster_mask']
        if args.device == 'cuda':
            audio = audio.to(args.device, non_blocking=True)
            text = text.to(args.device, non_blocking=True)
            if torch.is_tensor(text_mask):
                text_mask = text_mask.to(args.device, non_blocking=True)
            if torch.is_tensor(cluster_mask):
                cluster_mask = cluster_mask.to(args.device, non_blocking=True)
       
        # compute output
        loss, _, _, logit_scale = model(audio=audio, text=text, text_mask=text_mask, cluster_mask=cluster_mask)
        # train_losses.step(loss.item(), audio.size(0))
        
        if args.log:
            # logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
            # logger.log_logitscale(logit_scale, epoch * iters_per_epoch + data_iter_step)
            # logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/logit_scale": logit_scale.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if data_iter_step % args.print_freq == 0:
        #     progress.display(data_iter_step)

def validate(val_loader, model, epoch, args):
    # losses_val = AverageMeter('Valid Loss', ':.4e')
    # progress_val = ProgressMeter(len(val_loader),[losses_val],prefix="Epoch: [{}]".format(epoch))
    model.eval()
    epoch_end_loss, audio_corrects, text_corrects, cluster_count = [], [], [], []
    for data_iter_step, batch in enumerate(tqdm(val_loader)):
        audio = batch['audio']
        text = batch['text']
        text_mask = batch['text_mask']
        cluster_mask = batch['cluster_mask']
        cluster_count.append(cluster_mask.sum(dim=0))
        if args.device == 'cuda':
            audio = audio.to(args.device, non_blocking=True)
            text = text.to(args.device, non_blocking=True)
            if torch.is_tensor(text_mask):
                text_mask = text_mask.to(args.device, non_blocking=True)
            if torch.is_tensor(cluster_mask):
                cluster_mask = cluster_mask.to(args.device, non_blocking=True)
        with torch.no_grad():
            loss, audio_correct, text_correct, _ = model(audio, text, text_mask, cluster_mask)
        epoch_end_loss.append(loss.detach().cpu())
        audio_corrects.append(audio_correct.detach().cpu())
        text_corrects.append(text_correct.detach().cpu())
        # losses_val.step(loss.item(), audio.size(0))
        # if data_iter_step % args.print_freq == 0:
            # progress_val.display(data_iter_step)
    
    val_loss = torch.stack(epoch_end_loss).mean(0, False)
    audio_corrects = torch.stack(audio_corrects).sum(0, False)
    text_corrects = torch.stack(text_corrects).sum(0, False)
    if args.disentangle:
        cluster_count = torch.stack(cluster_count).sum(dim=0)
        weight = cluster_count / cluster_count.sum()
        audio_acc = audio_corrects / cluster_count
        text_acc = text_corrects / cluster_count
        avg_audio_acc = audio_acc.mean()
        avg_text_acc = text_acc.mean()
        wavg_audio_acc = (audio_acc * weight).sum()
        wavg_text_acc = (text_acc * weight).sum()
        audio_acc = torch.cat([audio_acc, avg_audio_acc.unsqueeze(0), wavg_audio_acc.unsqueeze(0)])
        text_acc = torch.cat([text_acc, avg_text_acc.unsqueeze(0), wavg_text_acc.unsqueeze(0)])
    else:
        audio_acc /= len(val_loader.dataset)
        text_acc /= len(val_loader.dataset)
    
    return val_loss, audio_acc, text_acc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    main()