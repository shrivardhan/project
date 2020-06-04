import json
import os
import subprocess
import pickle

coco_path = "./Dataset/COCOFeatures/"
coco_desc_path = "./Dataset/coco_train_captions.json"

f = open(coco_desc_path,"r")
jsonfile = json.load(f)
f.close()
annotation = jsonfile['annotations']
l = []
for image_id in os.listdir(coco_path):
    for i in range(len(annotation)):
        if annotation[i]['image_id']==int(image_id):
            while annotation[i]['image_id']==int(image_id):
                l.append(annotation[i])
                i = i+1
            break


'''for annot in jsonfile['annotations']:
    if str(annot['image_id']) not in os.listdir(coco_path):
        print("hello")'''

f = open("coco_annot.pkl","wb")
pickle.dump(l,f)
f.close()
