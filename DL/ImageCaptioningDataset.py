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

class ImageCaptioningDataset(object):
    def __init__(self,imagepath,coco_imagepath,image_desc_path,story_path,coco_desc_path,threshold=5):
        self.image_path = imagepath
        self.coco_image_path = coco_imagepath
        self.image_desc_map = image_desc_path
        self.coco_desc_path = coco_desc_path
        self.story_path = story_path
        self.image_desc = None
        self.coco_desc = None
        self.max_length = 2
        self.preprocess_data()
        self.image_path_map = self.get_image_path_map()
        self.vocabulary = self.build_vocabulary(threshold)
    
    def __len__(self):
        return len(self.coco_desc)+len(self.image_path_map.keys())
    
    def get_vocab_size(self):
        return self.vocabulary.get_size()
    
    def get_vocab(self):
        return self.vocabulary
    
    def get_image_desc_map(self):
        return self.image_desc

    def __getitem__(self,idx):
        image_ids = list(self.image_path_map.keys())
        if idx>=len(image_ids):
            idx = idx-len(image_ids)
            annot = self.coco_desc[idx]
            image_name = str(annot['image_id'])
            image = torch.load(self.coco_image_path+image_name)
            desc = annot['caption']
        else:
            id = image_ids[idx]
            image = torch.load(self.image_path_map[id])
            desc = self.image_desc[id]
        tokens = nltk.tokenize.word_tokenize(desc.lower())
        caption = []
        caption.append(self.vocabulary('<start>'))
        caption.extend([self.vocabulary(token) for token in tokens])
        caption.append(self.vocabulary('<end>'))
        desc = torch.Tensor(caption)
        return image,desc
    
    def collate_fn(data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]        
        return images, targets, lengths
    
    def preprocess_data(self):
        for file in os.listdir(self.image_path):
            if not os.path.isdir(self.image_path+file):
                os.remove(self.image_path+file)
        f = open(self.story_path,"rb")
        self.story_data = pickle.load(f)
        f.close()
        f = open(self.image_desc_map,"rb")
        self.image_desc = pickle.load(f)
        f.close()
        f = open(self.coco_desc_path,"rb")
        desc = pickle.load(f)
        self.coco_desc = desc
        
    
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
        '''if os.path.exists("./Dataset/vocabulary.pkl"):
            f = open("./Dataset/vocabulary.pkl","rb")
            vocabulary = pickle.load(f)
            return vocabulary'''
        vocabulary = Vocabulary()
        counter = Counter()
        for id in self.image_desc:
            caption = self.image_desc[id]
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            if len(tokens)>self.max_length-2:
                self.max_length = len(tokens)+2
            counter.update(tokens)
        
        for annot in self.coco_desc:
            caption = annot['caption']
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            if len(tokens)>self.max_length-2:
                self.max_length = len(tokens)+2
            counter.update(tokens)
        
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
