import os
import pickle
import shutil
import torch
import copy
from Vocabulary import Vocabulary
from collections import Counter

def preprocess_vist_data(): 
    image_path = "./Dataset/Images/"
    f = open("./Dataset/image_description.pkl","rb")
    image_description = pickle.load(f)
    f.close()
    f = open("./Dataset/story_annotation.pkl","rb")
    stories = pickle.load(f)
    f.close()
    temp = copy.deepcopy(stories)
    for story in stories:
        sequence = temp[story]
        for image in sequence:
            _,id,story_desc = image
            if id not in image_description.keys():
                if os.path.exists(image_path+story):
                    shutil.rmtree(image_path+story)
                if story in temp.keys():
                    del temp[story]

    for story in stories:
        if story in temp.keys():
            sequence = temp[story]
            for image in sequence:
                _,id,story_desc = image
                image_desc = image_description[id]
                if not os.path.exists(image_path+story+"/"+id):
                    if os.path.exists(image_path+story):
                        shutil.rmtree(image_path+story)
                    if story in temp.keys():
                        del temp[story]

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#preprocess_vist_data()