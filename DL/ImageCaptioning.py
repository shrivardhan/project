import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from ImageCaptioningDataset import ImageCaptioningDataset
from torch.nn.utils.rnn import pack_padded_sequence
from ImageCaptionModel import Encoder,Decoder

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

dataset = ImageCaptioningDataset(image_path,coco_image_path,image_desc_path,story_path,coco_desc_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,collate_fn=ImageCaptioningDataset.collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(feature_size,embed_size).to(device)
decoder = Decoder(embed_size,hidden_size, dataset.get_vocab_size(),num_layers).to(device)

#encoder = torch.load(model_path+'encoder.ckpt',map_location=torch.device(device))
#decoder = decoder.load_state_dict(torch.load(model_path+"decoder.pkl"))

params = list(decoder.parameters()) + list(encoder.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params, lr=learning_rate)

total_step = len(data_loader)
for epoch in range(num_epochs):
    i = 0
    for (images,captions,lengths) in data_loader:
        images = images.to(device)
        captions = captions.to(device)
        feature = encoder(images)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        outputs = decoder(feature,captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()


        if i%100==0:
            # Print log info
            print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, num_epochs, loss.item(), np.exp(loss.item()))) 

        if i%100==0: 
            # Save the model checkpoints
            torch.save(decoder, model_path+'decoder.ckpt')
            torch.save(encoder, model_path+'encoder.ckpt')
        i = i+1
