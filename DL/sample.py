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
model_path = './Models/'
image_path = './Dataset/Images/'
image_desc_path = './Dataset/image_description.pkl'
story_path = './Dataset/story_annotation.pkl'
coco_desc_path = "./Dataset/coco_annot.pkl"
coco_image_path = "./Dataset/COCOFeatures/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ImageCaptioningDataset(image_path,coco_image_path,image_desc_path,story_path,coco_desc_path)
image_desc = dataset.get_image_desc_map()
'''vocab = dataset.get_vocab()
encoder = Encoder(feature_size,embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder = Decoder(embed_size,hidden_size, dataset.get_vocab_size(),num_layers).to(device).eval()
encoder = encoder.to(device)
decoder = decoder.to(device)

# Load the trained model parameters
encoder = torch.load(model_path+'encoder.ckpt',map_location=torch.device("cpu"))
decoder = torch.load(model_path+'decoder.ckpt',map_location=torch.device("cpu"))

encoder = encoder.eval()  # eval mode (batchnorm uses moving mean/variance)
decoder = decoder.eval()


# Prepare an image
#image = load_image(args.image, transform)
feature_path = '/Volumes/Seagate Bac/VIST_Dataset/images3/train/features/'
image = torch.load(feature_path+"6534230087")
image_tensor = image.reshape(1,2048,1).to(device)
print(image_tensor.shape)

# Generate an caption from the image
feature = encoder(image_tensor)
sampled_ids = decoder.sample(feature)
sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

# Convert word_ids to words
sampled_caption = []
for word_id in sampled_ids:
    word = vocab.idx2word[word_id]
    sampled_caption.append(word)
    if word == '<end>':
        break
sentence = ' '.join(sampled_caption)'''

# Print out the image and the generated caption
print(image_desc["4936753335"])
print(image_desc["4562801065"])
print(image_desc["4562806105"])
print(image_desc["4563429868"])
print(image_desc["4563429868"])
#image = Image.open(args.image)
#plt.imshow(np.asarray(image))