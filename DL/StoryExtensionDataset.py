import torch
import torch.utils
import torch.utils.data
import os
import pickle
import shutil
import nltk
from Util import *
from Vocabulary import Vocabulary

class StoryExtensionDataset(object):
    def __init__(self,story_path,threshold=5):
        self.story_path = story_path
        self.story_data = None
        self.max_length = 2
        self.preprocess_data()
        self.vocabulary = self.build_vocabulary(threshold)
        self.data = None
    
    def __len__(self):
        return len(self.image_path_map.keys())
    
    def get_vocab_size(self):
        return self.vocabulary.get_size()
    
    def get_vocab(self):
        return self.vocabulary

    def __getitem__(self,idx):
        for seq in self.story_data:
            print(seq)
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
    
    def build_vocabulary(self,threshold):
        vocabulary = Vocabulary()
        counter = Counter()
        for id in self.image_desc:
            caption = self.image_desc[id]
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
        return vocabulary