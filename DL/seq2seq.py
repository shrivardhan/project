import torch
import torch.nn as nn
import numpy as np
import os
import pickle
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
num_epochs = 50
BATCH_SIZE = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = Seq2SeqDataset(image_desc_path,story_path,coco_desc_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,collate_fn=Seq2SeqDataset.collate_fn)
vocab_inp_size = dataset.get_vocab_size()
vocab = dataset.get_vocab()
encoder = Encoder(vocab_inp_size, embed_size, hidden_size,BATCH_SIZE)
encoder.to(device)
decoder = Decoder(vocab_inp_size, embed_size, hidden_size, hidden_size, BATCH_SIZE)
decoder = decoder.to(device)
decoder = torch.load(model_path+"s2s_dec_small.pth")
encoder = torch.load(model_path+"s2s_enc_small.pth")
criterion = nn.CrossEntropyLoss()

def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                       lr=0.001)

loss_list = []
perplexity = []
for epoch in range(num_epochs):
    for i,(input_seq,output,length) in enumerate(data_loader):
        enc_output,enc_hidden = encoder(input_seq.to(device),length,device)
        dec_hidden = enc_hidden#.squeeze(0)
        dec_input = torch.LongTensor([[int(vocab('<start>'))]] * BATCH_SIZE)
        loss = 0.0
        total_loss = 0
        for t in range(1, output.size(1)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                            dec_hidden.to(device), 
                                            enc_output.to(device))
            loss += loss_function(output[:, t].to(device), predictions.to(device))
                #loss += loss_
            dec_input = output[:, t].unsqueeze(1)
                
        batch_loss = (loss / int(output.size(1)))
        total_loss += batch_loss
        
        optimizer.zero_grad()
        
        loss.backward()

        loss_list.append(batch_loss.detach().item())
        perplexity.append(np.exp(batch_loss.detach().item()))
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        if i % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                        i,
                                                        batch_loss.detach().item()))
            torch.save(encoder,"Models/s2s_enc_small.pth")
            torch.save(decoder,"Models/s2s_dec_small.pth")

'''f = open("Models/train_s2s_loss.pkl","wb")
pickle.dump(loss_list,f)
f.close()
f = open("Models/train_s2s_perplexity.pkl","wb")
pickle.dump(perplexity,f)
f.close()'''

