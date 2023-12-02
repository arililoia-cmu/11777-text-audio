import os
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from config import CLUSTERS

class ECALS_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks,
                 text_preprocessor=None, text_type="bert", text_rep="stochastic",
                 disentangle=False, subset=False, test_mode='query'):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.disentangle = disentangle
        self.subset = subset
        self.test_mode = test_mode
        
        self.get_split()
        self.text_preprocessor = text_preprocessor
        self.text_type = text_type
        self.text_rep = text_rep
        print(f"Dataset: {self.split} split, {len(self.songs)} songs")
    
    def get_split(self):
        if self.subset:
            path = os.path.join(self.data_path, 'small')
        else:
            path = os.path.join(self.data_path, 'full')
        split_path = os.path.join(path, 'split.json')
        if self.disentangle:
            song_tag_path = os.path.join(path, 'clustered_song_tags.json')
            tag_path = os.path.join(path, 'clustered_tags.json')
            split_tag_path = os.path.join(path, 'clustered_split_tags.json')
        else:
            song_tag_path = os.path.join(path, 'song_tags.json')
            tag_path = os.path.join(path, 'all_tags.json')
            split_tag_path = os.path.join(path, 'split_tags.json')
        
        with open(split_path, 'r') as f:
            split = json.load(f)
            if self.split == 'TRAIN':
                if self.subset:
                    track = split['train_track']
                else:
                    track = split['train_track'] + split['extra_track']
            elif self.split == 'VALID':
                track = split['valid_track']
            elif self.split == 'TEST':
                track = split['test_track']
            else:
                raise ValueError(f'Unexpected split name: {self.split}')
            
        with open(song_tag_path, 'r') as f:
            song_tags = json.load(f)
            self.songs = [(k, song_tags[k]) for k in track]
        
        with open(tag_path, 'r') as f:
            tags = json.load(f)
            self.tags = tags
            if self.disentangle:
                self.tag_to_idx = dict()
                for cluster in CLUSTERS:
                    self.tag_to_idx[cluster] = {i:idx for idx, i in enumerate(self.tags[cluster])}
            else:
                self.tag_to_idx = {i:idx for idx, i in enumerate(self.tags)}
        
        with open(split_tag_path, 'r') as f:
            split_tags = json.load(f)
            if self.split == "TRAIN":
                if self.subset:
                    self.split_tags = split_tags['train_track']
                else:
                    self.split_tags = split_tags['train_track'] + split_tags['extra_track']
            elif self.split == "VALID":
                self.split_tags = split_tags['valid_track']
            elif self.split == "TEST":
                self.split_tags = split_tags['test_track']
        
        del split, song_tags, tags, split_tags
    
    def load_audio(self, msd_id, train=True):
        path = os.path.join(self.data_path, 'npy', msd_id + '.npy')
        audio = np.load(path, mmap_mode='r')
        if train:
            idx = np.random.randint(0, audio.shape[-1] - self.input_length)
            audio = np.array(audio[idx:idx+self.input_length])
        else:
            hop = (len(audio) - self.input_length) // self.num_chunks
            audio = np.stack(
                [np.array(audio[i * hop : i * hop + self.input_length]) 
                 for i in range(self.num_chunks)]
            )
        audio = torch.from_numpy(audio)
        return audio

    def load_text(self, tag_list):
        """
        input:  tag_list = list of tag
        output: text = string of text
        """
        if self.text_rep == "caption":
            if self.split == "TRAIN":
                random.shuffle(tag_list)
            text = tag_list
        elif self.text_rep == "tag":
            text = [random.choice(tag_list)]
        elif self.text_rep == "stochastic":
            k = random.randint(1, len(tag_list)) 
            text = random.sample(tag_list, k)
        return text
            
    def load_text_cluster(self, tag_list):
        """
        input: tag_list = list of tag
        output: texts = dict of cluster to string of text
                cluster_mask = binary mask of cluster
        """
        texts = dict()
        cluster_mask = []
        for cluster in CLUSTERS:
            if cluster not in tag_list:
                cluster_mask.append(0)
                texts[cluster] = ''
                continue
            if self.text_rep == "caption":
                if self.split == "TRAIN":
                    random.shuffle(tag_list[cluster])
                text = tag_list[cluster]
            elif self.text_rep == "tag":
                text = [random.choice(tag_list[cluster])]
            elif self.text_rep == "stochastic":
                k = random.randint(1, len(tag_list[cluster])) 
                text = random.sample(tag_list[cluster], k)
            texts[cluster] = text
            cluster_mask.append(1)
        return texts, np.array(cluster_mask)
    
    def tag_to_binary(self, tag_list):
        binary = np.zeros([len(self.tags),], dtype=np.float32)
        for tag in tag_list:
            binary[self.tag_to_idx[tag]] = 1.0
        return binary
    
    def tag_cluster_to_binary(self, tag_cluster):
        binary = dict()
        for cluster in CLUSTERS:
            binary[cluster] = np.zeros([len(self.tags[cluster]),], dtype=np.float32)
            if cluster in tag_cluster:
                for tag in tag_cluster[cluster]:
                    binary[cluster][self.tag_to_idx[cluster][tag]] = 1.0
        return binary
    
    def get_train_item(self, index):
        song = self.songs[index]
        tag_list = song[-1]
        if self.disentangle:
            text, cluser_mask = self.load_text_cluster(tag_list)
        else:
            text, cluser_mask = self.load_text(tag_list), None
        audio_tensor = self.load_audio(song[0])
        # audio_tensor = torch.rand(self.input_length)
        return {
            "audio":audio_tensor,
            "text":text,
            "cluster_mask":cluser_mask
        }

    def get_eval_item(self, index):
        song = self.songs[index]
        tag_list = song[-1]
        binary = self.tag_to_binary(tag_list)
        text = self.load_text(tag_list)
        tags = self.tags
        track_id = song[0]
        audio = self.load_audio(song[0], train=False)
        # audio = torch.rand((self.num_chunks, self.input_length))
        return {
            "audio":audio, 
            "track_id":track_id, 
            "tags":tags, 
            "binary":binary, 
            "text":text
        }

    def __getitem__(self, index):
        if self.split=='TRAIN' or self.split=='VALID' or self.test_mode=='loss':
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)

    def batch_processor(self, batch):
        # batch = list of dictionary
        audio = [item_dict['audio'] for item_dict in batch]
        audios = torch.stack(audio)
        if self.disentangle:
            # binary = dict()
            # for cluster in CLUSTERS:
            #     binary[cluster] = [item['binary'][cluster] for item in batch]
            # binarys = dict()
            # for cluster in CLUSTERS:
            #     binarys[cluster] = torch.tensor(np.stack(binary[cluster]))
            cluster_mask = torch.tensor(np.stack([item['cluster_mask'] for item in batch]))
        else:
            # binary = [item_dict['binary'] for item_dict in batch]
            # binarys = torch.tensor(np.stack(binary))
            cluster_mask = None
        text, text_mask = self._text_preprocessor(batch, "text")
        return {"audio":audios, "binary": None, "text":text, "text_mask":text_mask, "cluster_mask":cluster_mask}
    
    def _text_preprocessor(self, batch, target_text):
        if self.text_type == "bert":
            if self.disentangle:
                batch_text = [", ".join(item[target_text][cluster]) for cluster in CLUSTERS for item in batch]
                encoding = self.text_preprocessor.batch_encode_plus(batch_text, padding='longest', max_length=64, truncation=True, return_tensors="pt")
                text = encoding['input_ids']
                text_mask = encoding['attention_mask']
            else:
                batch_text = [", ".join(item_dict[target_text]) for item_dict in batch]
                encoding = self.text_preprocessor.batch_encode_plus(batch_text, padding='longest', max_length=64, truncation=True, return_tensors="pt")
                text = encoding['input_ids']
                text_mask = encoding['attention_mask']
        elif self.text_type == "glove":
            batch_emb = []
            batch_text = [item_dict[target_text] for item_dict in batch]
            for tag_seq in batch_text:
                tag_seq_emb = [np.array(self.text_preprocessor[token]).astype('float32') for token in tag_seq]
                batch_emb.append(torch.from_numpy(np.mean(tag_seq_emb, axis=0)))
            text = torch.stack(batch_emb)
            text_mask = None
        return text, text_mask
            
    def __len__(self):
        return len(self.songs)
