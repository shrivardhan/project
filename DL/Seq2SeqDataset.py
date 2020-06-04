import torch
import torch.utils
import torch.utils.data
import os
import json
import pickle
import shutil
import nltk
from Util import *
from Vocabulary import Vocabulary

class Seq2SeqDataset(object):
    def __init__(self,image_desc_path,story_path,coco_desc_path,threshold=5):
        self.image_desc_map = image_desc_path
        self.story_path = story_path
        self.coco_desc_path = coco_desc_path
        self.input_maxlen = 2
        self.output_maxlen = 2
        self.preprocess_data()
        self.vocabulary = self.build_vocabulary(threshold)
    
    def __len__(self):
        return 64
    
    def get_vocab_size(self):
        return self.vocabulary.get_size()
    
    def get_vocab(self):
        return self.vocabulary
    
    def get_image_desc_map(self):
        return self.image_des
    

    def __getitem__(self,idx):
        ids = list(self.story_data.keys())
        output_seq = []
        input_seq = []
        output_seq.append(self.vocabulary('<start>'))
        for seq in self.story_data[ids[idx]]:
            caption = seq[2]
            print(caption)
            tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(caption.lower())
            output_seq.extend([self.vocabulary(token) for token in tokens])
            image_id = seq[1]
            caption = self.image_desc[image_id]
            print(caption)
            tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(caption.lower())
            input_seq.extend([self.vocabulary(token) for token in tokens])
        output_seq.append(self.vocabulary('<end>'))
        input_seq.append(self.vocabulary('<end>'))
        return torch.Tensor(input_seq),torch.Tensor(output_seq)
    
    def collate_fn(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        input_seq,output_seq = zip(*data)
        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths_in = [len(cap) for cap in input_seq]
        input_tensor = torch.zeros(len(input_seq), max(lengths_in)).long()
        for i, cap in enumerate(input_seq):
            end = lengths_in[i]
            input_tensor[i, :end] = cap[:end]  
        lengths = [len(cap) for cap in output_seq]
        output_tensor = torch.zeros(len(output_seq), max(lengths)).long()
        for i, cap in enumerate(output_seq):
            end = lengths[i]
            output_tensor[i, :end] = cap[:end]
        return input_tensor,output_tensor, lengths_in
    
    def preprocess_data(self):
        f = open(self.story_path,"rb")
        self.story_data = pickle.load(f)
        f.close()
        f = open(self.image_desc_map,"rb")
        self.image_desc = pickle.load(f)
        f.close()
        f= open(self.coco_desc_path,"rb")
        self.coco_desc = pickle.load(f)
        f.close()
        processed_data = {}
        for id in self.story_data:
            flag=0
            for seq in self.story_data[id]:
                if seq[1] not in self.image_desc.keys():
                    flag=1
                    break
            if flag==0:
                processed_data[id] = self.story_data[id]
        
        self.story_data = processed_data

    
    def get_image_path_map(self):
        image_path_map = {}
        f = open(self.story_path,"rb")
        self.story_data = pickle.load(f)
        f.close()
        for id in os.listdir(self.image_path):
            sequence = self.story_data[id]
            for _,photo_id,story_desc in sequence:
                image_path_map[photo_id] = self.image_path+id+"/"+photo_id
        return image_path_map

    
    def build_vocabulary(self,threshold):
        vocabulary = Vocabulary()
        counter = Counter()
        for id in self.image_desc:
            caption = self.image_desc[id]
            tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(caption.lower())
            if len(tokens)>self.input_maxlen-2:
                self.input_maxlen = len(tokens)+2
            counter.update(tokens)
        
        for id in self.story_data:
            temp_in = 0
            temp_out = 0
            for seq in self.story_data[id]:
                caption = seq[2]
                tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(caption.lower())
                counter.update(tokens)
                temp_out = temp_out+len(tokens)
                caption_in = self.image_desc[seq[1]]
                tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(caption_in.lower())
                temp_in = temp_in +len(tokens)
                counter.update(tokens)
            if temp_out>self.output_maxlen-2:
                self.output_maxlen = temp_out+2
            if temp_in>self.input_maxlen-2:
                self.input_maxlen = temp_out+2
        
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        # Create a vocab wrapper and add some special tokens.
        vocabulary = Vocabulary()
        vocabulary.add_word('<pad>')
        vocabulary.add_word('<start>')
        vocabulary.add_word('<end>')
        vocabulary.add_word('<unk>')

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocabulary.add_word(word)
        

        f = open("./Dataset/vocabulary.pkl","wb")
        pickle.dump(vocabulary,f)
        f.close()
        return vocabulary
