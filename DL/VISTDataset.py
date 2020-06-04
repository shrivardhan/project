import torch
import torch.utils
import torch.utils.data
import os
import pickle
import shutil
import nltk
from Util import *
from Vocabulary import Vocabulary

class Sequence(object):
    def __init__(self,sequence):
        self.sequence = sequence

class VISTDataset(object):
    def __init__(self,imagepath,image_desc_path,story_path,threshold=5):
        self.image_path = imagepath
        self.image_desc_map = image_desc_path
        self.story_path = story_path
        self.story_data = None
        self.image_desc = None
        self.preprocessData()
        self.data = self.loadData()
        self.vocabulary = self.buildVocabulary(threshold)
        self.maxlength = 0
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        image_sequence = []
        desc_sequence = []
        story_sequence = [] 
        l = []
        for data in self.data[idx].sequence:
            image_sequence.append(data[0])
            desc_sequence.append(data[1])
            story_sequence.append(data[2])
        return image_sequence,desc_sequence,story_sequence
    
    def loadData(self):
        data = []
        image_folders = os.listdir(self.image_path)
        for folder in image_folders:
            story_id = folder
            sequence = self.story_data[story_id]
            story = []
            count = 0
            for image in sequence:
                _,photo_id,story_desc = image
                desc = self.image_desc[photo_id]
                tokens = nltk.tokenize.word_tokenize(desc.lower())
                tokens = nltk.tokenize.word_tokenize(str(desc).lower())
                caption = []
                caption.append(vocab('<start>'))
                caption.extend([self.vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                desc = torch.Tensor(caption)
                tokens = nltk.tokenize.word_tokenize(story_desc.lower())
                tokens = nltk.tokenize.word_tokenize(str(story_desc).lower())
                caption = []
                caption.append(vocab('<start>'))
                caption.extend([self.vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                story_desc = torch.Tensor(caption)
                image_feature = torch.load(self.image_path+story_id+"/"+photo_id)
                story.append((image_feature,story_desc,desc))
            data.append(obj)
        return data
    
    def preprocessData(self):
        for file in os.listdir(self.image_path):
            if not os.path.isdir(self.image_path+file):
                os.remove(self.image_path+file)
        f = open(self.story_path,"rb")
        self.story_data = pickle.load(f)
        f.close()
        f = open(self.image_desc_map,"rb")
        self.image_desc = pickle.load(f)
        f.close()
    
    def buildVocabulary(self,threshold):
        vocabulary = Vocabulary()
        counter = Counter()
        for id in self.image_desc:
            caption = self.image_desc[id]
            if len(caption)>self.maxlength:
                self.maxlength = len(caption) 
            tokens = nltk.tokenize.word_tokenize(caption.lower())
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


dataset = VISTDataset('./Dataset/Images/','./Dataset/image_description.pkl','./Dataset/story_annotation.pkl')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for image,desc in data_loader:
    print