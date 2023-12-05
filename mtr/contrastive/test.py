import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from transformers import AutoModel, AutoTokenizer, set_seed
from tqdm import tqdm
import json
import pandas as pd
import time

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from dataset import ECALS_Dataset
from config import get_parser, CLUSTERS
from model import ContrastiveModel
from mtr.modules.audio_rep import TFRep
from mtr.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mtr.modules.encoder import MusicTransformer
from mtr.utils.eval_utils import single_query_evaluation, retrieval_evaluation, _text_representation

TAGNAMES = [
    'rock','pop','indie','alternative','electronic','hip hop','metal','jazz','punk',
    'folk','alternative rock','indie rock','dance','hard rock','00s','soul','hardcore',
    '80s','country','classic rock','punk rock','blues','chillout','experimental',
    'heavy metal','death metal','90s','reggae','progressive rock','ambient','acoustic',
    'beautiful','british','rnb','funk','metalcore','mellow','world','guitar','trance',
    'indie pop','christian','house','spanish','latin','psychedelic','electro','piano',
    '70s','progressive metal',
]


def main():
    print("Start Testing")
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

    main_worker(args)

def main_worker(args):
    t0 = time.time()
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
        attention_ndim= args.attention_ndim
    )

    if args.text_type == "bert":
        text_encoder = AutoModel.from_pretrained(args.text_backbone)
        tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
        args.text_dim = 768
    elif args.text_type == "glove":
        text_encoder = nn.Identity()
        tokenizer = torch.load(os.path.join(args.data_path, "ecals_annotation", "glove_tag_embs.pt"))
        args.text_dim = 300

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
        n_proj = args.n_proj,
        combine = args.combine,
        audio_w = args.audio_w
    )
    save_dir = args.save_path
    assert args.name is not None, "Please specify the model name"
    pretrained_object = torch.load(f'{save_dir}/{args.name}.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    
    model = model.to(args.device)
    cudnn.benchmark = True
    model.eval()

    test_dataset= ECALS_Dataset(
        data_path = args.data_path,
        split = "TEST",
        sr = args.sr,
        duration = args.duration,
        num_chunks = args.num_chunks,
        text_preprocessor= tokenizer,
        text_type=args.text_type,
        text_rep = args.text_rep,
        disentangle = args.disentangle,
        subset = args.subset,
        test_mode = args.test
    )

    if args.test == 'loss':
        loss(test_dataset, model, args)

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
            subset = args.subset,
            test_mode = args.test
        )
        loss(val_dataset, model, args)
    else:
        if args.emb:
            embeddings, tag_embs = torch.load(os.path.join(save_dir, f"{args.emb}_emb.pt"))
        else:
            embeddings, tag_embs = get_embeddings(args, model, test_dataset, save_dir, tokenizer)
            torch.save([embeddings, tag_embs], os.path.join(save_dir, f"{args.model}_emb.pt"))
        single_query(args, test_dataset, embeddings, tag_embs, save_dir)
        retrieval(args, embeddings, save_dir)

def get_embeddings(args, model, test_dataset, tokenizer):
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None, collate_fn=test_dataset.eval_batch,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    embeddings = dict()
    count = 0
    for batch in tqdm(test_loader):
        track_id = batch['track_id']
        
        # (Batch, Segment, Length) -> (Batch * Segment, Length)
        audio = batch['audio']
        audio = audio.reshape(-1, audio.shape[-1])
        token = batch['token']
        
        if args.device != 'cpu':
            audio = audio.to(args.device, non_blocking=True)
            token = token.to(args.device, non_blocking=True)
        with torch.no_grad():
            # ((N_clusters), Batch * Segment, Dim) -> ((N_clusters), Batch, Dim)
            z_audio = model.encode_audio(audio).detach().cpu()
            if args.disentangle:
                z_audio = z_audio.reshape(len(CLUSTERS), len(track_id), -1, z_audio.shape[-1])
                z_audio = z_audio.mean(2)
            else:
                z_audio = z_audio.reshape(len(track_id), -1, z_audio.shape[-1])
                z_audio = z_audio.mean(1)
            if args.text_type == "bert":
                z_caption = model.encode_bert_text(token, None).detach().cpu()
            elif args.text_type == "glove":
                z_caption = model.encode_glove_text(token).detach().cpu()
            if args.disentangle:  # (N_clusters, Batch, Dim) -> (Batch, N_clusters, Dim)
                z_audio = z_audio.permute(1,0,2)
                z_caption = z_caption.permute(1,0,2)
        
        for i, tid in enumerate(track_id):
            embeddings[tid] = {
                "z_audio": z_audio[i],
                "track_id": tid,
                "text": batch['text'][i],
                "binary": batch['binary'][i],
                "z_text": z_caption[i],
            }
            if args.disentangle:
                embeddings[tid]['cluster_mask'] = batch['cluster_mask'][i]
        count += 1
        if count == 5:
            break
    
    if args.disentangle:
        tag_embs = dict()
        for c in CLUSTERS:
            embs = []
            for tag in tqdm(test_dataset.tags[c], desc=c):
                input_text = _text_representation(args, tag, tokenizer)
                if args.gpu is not None:
                    input_text = input_text.to(args.device, non_blocking=True)
                with torch.no_grad():
                    z_tag = model.encode_bert_tag(input_text, c)
                embs.append(z_tag.detach().cpu())
            tag_embs[c] = embs
    else:
        tag_embs = []
        for tag in tqdm(test_dataset.tags):
            input_text = _text_representation(args, tag, tokenizer)
            if args.gpu is not None:
                input_text = input_text.to(args.device, non_blocking=True)
            with torch.no_grad():
                z_tag = model.encode_bert_text(input_text, None)
            tag_embs.append(z_tag.detach().cpu())
    
    return embeddings, tag_embs

def single_query(args, test_dataset, embeddings, all_tag_embs, save_dir):
    all_audio_embs = torch.stack([v['z_audio'] for v in embeddings.values()])
    all_tags = [v['text'] for v in embeddings.values()]
    
    if args.disentangle:
        results = dict()
        for i, c in enumerate(CLUSTERS):
            ids = torch.tensor([j for j in range(len(all_tags)) if all_tags[j][c]], dtype=torch.long)
            audio_embs = all_audio_embs[ids, i]
            audio_embs = nn.functional.normalize(audio_embs, dim=1)
            tag_embs = torch.cat(all_tag_embs[c], dim=0)
            tag_embs = nn.functional.normalize(tag_embs, dim=1)
            targets = np.stack([v['binary'][c] for v in embeddings.values()])

            logits = audio_embs @ tag_embs.T
            logits_df = pd.DataFrame(logits.numpy(), index=embeddings.keys(), columns=test_dataset.tags[c])
            targets_df = pd.DataFrame(targets, index=embeddings.keys(), columns=test_dataset.tags[c])
            
            result = single_query_evaluation(targets_df, logits_df, test_dataset.tags[c])
            results[c] = result
        json.dump(results, open(os.path.join(save_dir, "cluster_results.json"),'w'), indent=4)
    
    else:
        audio_embs = nn.functional.normalize(all_audio_embs, dim=1)
        tag_embs = torch.cat(tag_embs, dim=0)
        tag_embs = nn.functional.normalize(tag_embs, dim=1)
        targets = np.stack([v['binary'] for v in embeddings.values()])

        logits = audio_embs @ tag_embs.T
        logits_df = pd.DataFrame(logits.numpy(), index=embeddings.keys(), columns=test_dataset.tags)
        targets_df = pd.DataFrame(targets, index=embeddings.keys(), columns=test_dataset.tags)

        results = single_query_evaluation(targets_df, logits_df, save_dir, TAGNAMES)
        json.dump(results, open(os.path.join(save_dir, f"{args.name}_{len(TAGNAMES)}_results.json"),'w'), indent=4)
        
        single_query_evaluation(targets_df, logits_df, save_dir, test_dataset.split_tags)
        json.dump(results, open(os.path.join(save_dir, f"{args.name}_{len(test_dataset.split_tags)}_results.json"),'w'), indent=4)


def retrieval(args, embeddings, save_dir):
    retrieval_query = json.load(open(os.path.join(args.data_path, "retrieval_query.json"),'r'))
    embeddings = {k:v for k,v in embeddings.items() if k in retrieval_query.keys()}

    if args.disentangle:
        cluster_mask = torch.tensor(np.array([v['cluster_mask'] for v in embeddings.values()]))
        
        audio_embs = torch.stack([v['z_audio'] for v in embeddings.values()])
        audio_embs = audio_embs.view(audio_embs.shape[0], -1)
        audio_embs = nn.functional.normalize(audio_embs, dim=1)
        
        text_embs = torch.stack([v['z_text'] for v in embeddings.values()], dim=0)
        text_embs *= cluster_mask.unsqueeze(-1)  # Mask out non-existing clusters
        text_embs = text_embs.view(text_embs.shape[0], -1)
        text_embs = nn.functional.normalize(text_embs, dim=1)

        captions, songs = [], []
        for k, v in embeddings.items():
            caption = []
            for c in CLUSTERS:
                caption.extend(v['text'][c])
            caption = ", ".join(caption)
            captions.append(caption)
            songs.append(k)
    else:
        audio_embs = torch.stack([v['z_audio'] for v in embeddings.values()])
        audio_embs = nn.functional.normalize(audio_embs, dim=1)
        
        text_embs = torch.stack([v['z_text'] for v in embeddings.values()], dim=0)
        text_embs = nn.functional.normalize(text_embs, dim=1)

        captions = [", ".join(v['text']) for v in embeddings.values()]
        songs = [k for k in embeddings.keys()]
    
    results = retrieval_evaluation(audio_embs, text_embs, captions, songs)
    json.dump(results, open(os.path.join(save_dir, "{args.name}_retrieval_results.json"),'w'), indent=4)
        


def loss(dataset, model, args):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=None, collate_fn=dataset.batch_processor,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
    epoch_end_loss, audio_corrects, text_corrects, cluster_count = [], [], [], []
    for batch in tqdm(loader):
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
        print(f"Loss: {val_loss}")
        print(f"Audio Acc: {avg_audio_acc}, Text Acc: {avg_text_acc}")
        print(f"Weighted Audio Acc: {wavg_audio_acc}, Weighted Text Acc: {wavg_text_acc}")
        for i, c in enumerate(CLUSTERS):
            print(f"{c}: Audio Acc: {audio_acc[i]}, Text Acc: {text_acc[i]}")
    else:
        audio_acc /= len(loader.dataset)
        text_acc /= len(loader.dataset)
        print(f"Loss: {val_loss}")
        print(f"Audio Acc: {audio_acc}, Text Acc: {text_acc}")

    return val_loss, audio_acc, text_acc

if __name__ == '__main__':
    main()