import json
import os
import subprocess
import pickle

image_path = "/Volumes/Seagate Bac/VIST_Dataset/images3/train/"
path = "/Volumes/Seagate Bac/VIST_Processed/"
files = os.listdir(image_path)
for i in range(len(files)):
    file = files[i][:-4]
    files[i] = file
description_file  = open("Dataset/dii/train.description-in-isolation.json","r")
story_file = open("Dataset/sis/train.story-in-sequence.json","r")
desc_data= json.load(description_file)
story_data = json.load(story_file)
descriptions = []
stories = {}
images_file = []
image_to_desc = {}
for annotation in story_data['annotations']:
    print(annotation[0].keys())
    order = annotation[0]['worker_arranged_photo_order']
    story_text = annotation[0]['text']
    photo_id = annotation[0]['photo_flickr_id']
    story_id = annotation[0]['story_id']
    if not story_id in stories.keys():
        stories[story_id] = []
    stories[story_id].append((order,photo_id,story_text))
    stories[story_id].sort(key = lambda x:x[0])
for annotation in desc_data['annotations']:
    description_text = annotation[0]['text']
    photo_id = annotation[0]['photo_flickr_id']
    image_to_desc[photo_id] = description_text

'''for story in stories:
    sequence = stories[story]
    for image in sequence:
        _,photo_id,story_text = image
        if not photo_id in image_to_desc.keys():
            break
        desc = image_to_desc[photo_id]
        if photo_id in files:
            cmd = ['mkdir',path+story]
            subprocess.call(cmd)
            cmd = ["cp",image_path+photo_id+".jpg",path+story+"/"+photo_id+".jpg"]
            subprocess.call(cmd)
            cmd = ["cp",image_path+"features/"+photo_id,path+story+"/"+photo_id]
            subprocess.call(cmd)
            cmd = ['mkdir',"./Dataset/Images/"+story]
            subprocess.call(cmd)
            cmd = ["cp",path+story+"/"+photo_id,"./Dataset/Images/"+story+"/"+photo_id] 
            subprocess.call(cmd)

f = open("./Dataset/story_annotation.pkl","wb")
pickle.dump(stories,f)
f.close()

f = open("./Dataset/image_description.pkl","wb")
pickle.dump(image_to_desc,f)
f.close()'''



