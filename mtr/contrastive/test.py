import os
import sys
from pathlib import Path
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
from config import get_parser
from model import ContrastiveModel
from mtr.modules.audio_rep import TFRep
from mtr.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mtr.modules.encoder import MusicTransformer
from mtr.utils.eval_utils import single_query_evaluation, multi_query_evaluation, _text_representation

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
        disentangle = args.disentangle
    )
    save_dir = f"exp/{args.text_type}_{args.text_rep}"
    # save_dir = f"exp/{args.arch}_{args.frontend}_{args.mix_type}_{args.audio_rep}/{args.text_type}_{args.text_rep}"
    # pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    # state_dict = pretrained_object['state_dict']
    # for k in list(state_dict.keys()):
    #     if k.startswith('module.'):
    #         state_dict[k[len("module."):]] = state_dict[k]
    #     del state_dict[k]
    # model.load_state_dict(state_dict, strict=False)

    
    model = model.to(args.device)
    cudnn.benchmark = True

    test_dataset= ECALS_Dataset(
        data_path = args.data_path,
        split = "TEST",
        sr = args.sr,
        duration = args.duration,
        num_chunks = args.num_chunks,
        text_preprocessor= tokenizer,
        text_type=args.text_type,
        text_rep = args.text_rep
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    multi_query_set = json.load(open(os.path.join(args.data_path, "multiquery_samples.json"),'r'))
    model.eval()
    track_ids, audio_embs, groudturths, audio_dict, multi_query_dict = [], [], [], {}, {}
    for batch in tqdm(test_loader):
        audio = batch['audio']
        list_of_tag = [tag[0] for tag in batch['text']]
        caption = ", ".join(list_of_tag)
        track_id = batch['track_id'][0]
        track_ids.append(track_id)
        groudturths.append(batch['binary'])
        input_text = _text_representation(args, list_of_tag, tokenizer)
        if args.gpu is not None:
            audio = audio.to(args.device, non_blocking=True)
            input_text = input_text.to(args.device, non_blocking=True)
        with torch.no_grad():
            z_audio = model.encode_audio(audio.squeeze(0))
            if args.text_type == "bert":
                z_caption = model.encode_bert_text(input_text, None)
            elif args.text_type == "glove":
                z_caption = model.encode_glove_text(input_text)
        if track_id in multi_query_set.keys():
            multi_query_dict[track_id] = {
                "z_audio": z_audio.mean(0).detach().cpu(),
                "track_id": track_id,
                "text": caption,
                "binary": batch['binary'],
                "z_text": z_caption.detach().cpu()
            }
        audio_embs.append(z_audio.mean(0).detach().cpu())
        audio_dict[track_id] = z_audio.mean(0).detach().cpu()
    audio_embs = torch.stack(audio_embs, dim=0)
    
    # single query evaluation
    tag_embs, tag_dict = [], {}
    for tag in test_dataset.tags:
        input_text = _text_representation(args, tag, tokenizer)
        if args.gpu is not None:
            input_text = input_text.to(args.device, non_blocking=True)
        with torch.no_grad():
            if args.text_type == "bert":
                z_tag = model.encode_bert_text(input_text, None)
            elif args.text_type == "glove":
                z_tag = model.encode_glove_text(input_text)
        tag_embs.append(z_tag.detach().cpu())
        tag_dict[tag] = z_tag.detach().cpu()

    torch.save(audio_dict, os.path.join(save_dir, "audio_embs.pt"))
    torch.save(tag_dict, os.path.join(save_dir, "tag_embs.pt"))
    torch.save(multi_query_dict, os.path.join(save_dir, "caption_embs.pt"))

    tag_embs = torch.cat(tag_embs, dim=0)
    targets = torch.cat(groudturths, dim=0)
    audio_embs = nn.functional.normalize(audio_embs, dim=1)
    tag_embs = nn.functional.normalize(tag_embs, dim=1)
    single_query_logits = audio_embs @ tag_embs.T

    sq_logits = pd.DataFrame(single_query_logits.numpy(), index=track_ids, columns=test_dataset.tags)
    sq_targets = pd.DataFrame(targets.numpy(), index=track_ids, columns=test_dataset.tags)

    single_query_evaluation(sq_targets, sq_logits, save_dir, TAGNAMES) # 50 tag evaluation
    single_query_evaluation(sq_targets, sq_logits, save_dir, test_dataset.split_tags) # test split tag evaluation
    # single_query_evaluation(sq_targets, sq_logits, save_dir, test_dataset.tags) # 1054 tag evaluation
    # multi_query_evaluation(tag_dict, multi_query_dict, save_dir) # multi_query evaluation


if __name__ == '__main__':
    main()