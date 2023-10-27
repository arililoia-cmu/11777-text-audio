import os
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class ECALS_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks,
                 text_preprocessor=None, text_type="bert", text_rep="stochastic"):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.get_split()
        self.text_preprocessor = text_preprocessor
        self.text_type = text_type
        self.text_rep = text_rep
        print(f"Dataset: {self.split} split, {len(self.songs)} songs")
    
    def get_split(self):
        split_path = os.path.join(self.data_path, 'split.json')
        song_tag_path = os.path.join(self.data_path, 'song_tags.json')
        ecals_tag_path = os.path.join(self.data_path, 'ecals_tags.json')
        split_tag_path = os.path.join(self.data_path, 'split_tags.json')
        
        with open(split_path, 'r') as f:
            split = json.load(f)
            if self.split == 'TRAIN':
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
        
        with open(ecals_tag_path, 'r') as f:
            ecals_tags = json.load(f)
            self.tags = ecals_tags
            self.tag_to_idx = {i:idx for idx, i in enumerate(self.tags)}
        
        with open(split_tag_path, 'r') as f:
            split_tags = json.load(f)
            if self.split == "TRAIN":
                self.split_tags = split_tags['train_track'] + split_tags['extra_track']
            elif self.split == "VALID":
                self.split_tags = split_tags['valid_track']
            elif self.split == "TEST":
                self.split_tags = split_tags['test_track']
        
        del split, song_tags, ecals_tags, split_tags
    
    def load_audio(self, msd_id, train=True):
        path = os.path.join(self.data_path, 'npy', msd_id + '.npy')
        audio = np.load(path, mmap_mode='r')
        if train:
            idx = np.random.randint(0, audio.shape[-1] - self.input_length)
            audio = audio[idx:idx+self.input_length]
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
    
    def tag_to_binary(self, tag_list):
        binary = np.zeros([len(self.tags),], dtype=np.float32)
        for tag in tag_list:
            binary[self.tag_to_idx[tag]] = 1.0
        return binary
    
    def get_train_item(self, index):
        song = self.songs[index]
        tag_list = song[-1]
        binary = self.tag_to_binary(tag_list)
        text = self.load_text(tag_list)
        audio_tensor = self.audio_load(song[0])
        return {
            "audio":audio_tensor, 
            "binary":binary, 
            "text":text
            }

    def get_eval_item(self, index):
        song = self.songs[index]
        tag_list = song[-1]
        binary = self.tag_to_binary(tag_list)
        text = self.load_text(tag_list)
        tags = self.tags
        track_id = song[0]
        audio = self.load_audio(track_id, train=False)
        return {
            "audio":audio, 
            "track_id":track_id, 
            "tags":tags, 
            "binary":binary, 
            "text":text
        }

    def __getitem__(self, index):
        if (self.split=='TRAIN') or (self.split=='VALID'):
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)

    def batch_processor(self, batch):
        # batch = list of dictionary
        audio = [item_dict['audio'] for item_dict in batch]
        binary = [item_dict['binary'] for item_dict in batch]
        audios = torch.stack(audio)
        binarys = torch.tensor(np.stack(binary))
        text, text_mask = self._text_preprocessor(batch, "text")
        return {"audio":audios, "binary":binarys, "text":text, "text_mask":text_mask}
    
    def _text_preprocessor(self, batch, target_text):
        if self.text_type == "bert":
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
