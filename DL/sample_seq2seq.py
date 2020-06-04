import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from Seq2SeqDataset import Seq2SeqDataset
from torch.nn.utils.rnn import pack_padded_sequence
from SeqEncoder import Encoder
from AttnSeqDecoder import Decoder

feature_size = 2048
embed_size = 256
num_layers = 1
hidden_size = 512
learning_rate = 1e-3
image_path = './Dataset/Images/'
image_desc_path = './Dataset/image_description.pkl'
story_path = './Dataset/story_annotation.pkl'
coco_desc_path = "./Dataset/coco_annot.pkl"
coco_image_path = "./Dataset/COCOFeatures/"
model_path = "./Models/"
num_epochs = 1000
BATCH_SIZE = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = Seq2SeqDataset(image_desc_path,story_path,coco_desc_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64,collate_fn=Seq2SeqDataset.collate_fn)
vocab_inp_size = dataset.get_vocab_size()
vocab = dataset.get_vocab()
encoder = Encoder(vocab_inp_size, embed_size, hidden_size,BATCH_SIZE)
encoder = encoder.to(device)
decoder = Decoder(vocab_inp_size, embed_size, hidden_size, hidden_size, BATCH_SIZE)
decoder = decoder.to(device)
decoder = torch.load(model_path+"s2s_dec_small.pth",map_location=torch.device("cpu"))
encoder = torch.load(model_path+"s2s_enc_small.pth",map_location=torch.device("cpu"))
criterion = nn.CrossEntropyLoss()
encoder.eval()
decoder.eval()
story = []
orig_story = []
for epoch in range(num_epochs):
    for i,(input_seq,output,length) in enumerate(data_loader):
        enc_output,enc_hidden = encoder(input_seq.to(device),length,device)
        dec_hidden = enc_hidden#.squeeze(0)
        dec_input = torch.LongTensor([[int(vocab('<start>'))]] * BATCH_SIZE)
        loss = 0.0
        total_loss = 0
        for t in range(1, output.size(1)):
            predictions, dec_hidden, attn = decoder(dec_input.to(device), 
                                            dec_hidden.to(device), 
                                            enc_output.to(device))
            dec_input = torch.argmax(predictions,dim=1)
            story.append(dec_input[4].cpu().item())
            orig_story.append(output[:,t][4].cpu().item())
            dec_input = dec_input.unsqueeze(1)
        break
    break

ans = ""
for id in story:
    ans = ans+" "+vocab.idx2word[id]
print(ans)

print("\n\n")
ans = ""
for id in orig_story:
    ans = ans+" "+vocab.idx2word[id]
print(ans)
