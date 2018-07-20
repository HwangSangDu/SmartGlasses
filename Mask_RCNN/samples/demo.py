
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


C_END     = "\033[0m"
C_BOLD    = "\033[1m"
C_INVERSE = "\033[7m"
 
C_BLACK  = "\033[30m"
C_RED    = "\033[31m"
C_GREEN  = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE   = "\033[34m"
C_PURPLE = "\033[35m"
C_CYAN   = "\033[36m"
C_WHITE  = "\033[37m"
 
C_BGBLACK  = "\033[40m"
C_BGRED    = "\033[41m"
C_BGGREEN  = "\033[42m"
C_BGYELLOW = "\033[43m"
C_BGBLUE   = "\033[44m"
C_BGPURPLE = "\033[45m"
C_BGCYAN   = "\033[46m"
C_BGWHITE  = "\033[47m"




def printComment(str):
  print(C_BOLD +  C_GREEN + str + C_END)
def printError(str):
  print(C_BOLD +  C_RED + str + C_END)


import re
import time
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


printComment("ROOT_DIR : " + ROOT_DIR)
printComment("IMAGE_DIR : " + IMAGE_DIR)
printComment("MODEL_DIR : " + MODEL_DIR)
printComment("COCO_MODEL_PATH : " + COCO_MODEL_PATH)



# ls
# return length
def search(dir):
        files = os.listdir(dir)
        # for file in files:
        #         print(file)
        return len(files)









# while True :

#   # 추가 시 길이에 변화가 생기면서 루프문을 벗어난다.
#   printComment("Length Checking...")
#   while (search(IMAGE_DIR)) == listLength:
#    time.sleep(0.1)  # 0.1초
#    printError("Time Sleep")
#    continue
#   # 길이를 업데이트한다.
#   printComment("Add New Image File")
#   listLength = search(IMAGE_DIR)
#   print(listLength)






# def search2(dir):
#         files = os.listdir(dir)
#         for file in files:
#                 fullFilename = os.path.join(dir, file)
#                 print(fullFilename)
# print()
# print()
# print()
# search2(IMAGE_DIR)




# def search3(dir):
#         files = os.listdir(dir)
#         for file in files:
#                 fullFilename = os.path.join(dir, file)
#                 if os.path.isdir(fullFilename):
#                         search(fullFilename)
#                 else:
#                         print(fullFilename)

# print()
# print()
# print()
# search3(IMAGE_DIR)









# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[1]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
printComment("Create Model Object")
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
printComment("LOAD MS-COCO")
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

# In[5]:

while True :

  listLength = search(IMAGE_DIR)
  print(listLength)
  # 추가 시 길이에 변화가 생기면서 루프문을 벗어난다.
  printComment("Length Checking...")
  while (search(IMAGE_DIR)) == listLength:
   time.sleep(0.1)  # 0.1초
   printError("Time Sleep")
   continue

  
  # 길이를 업데이트한다.
  printComment("Add New Image File")
  listLength = search(IMAGE_DIR)
  print(listLength)


  ## Excute Command 
  printComment("Remove /.DS_Store")
  #command = "rm " + IMAGE_DIR + "/*.json"
  #print(command)
  #os.system(command)
  command = "rm " + IMAGE_DIR + "/.DS_Store"
  printComment(command)
  os.system(command)
  print()


  ## Load a random image from the images folder
  printComment("Load Image")
  file_names = next(os.walk(IMAGE_DIR))[2]
  printComment( "Selected Image : " + file_names[0])
  print()

  ## file_names Filtering
  # file_names.extend(["1","2","3","4","5"])
  print(file_names)
  printComment("Filtering file_names")
  # p = re.compile('\.(jpg|gif|png)')
  p = re.compile('\w+\.(jpg|gif|png|jpeg)')


  file_names = list(filter(p.match, file_names)) # Read Note
  print(file_names)
  printComment("Regex Completion")
  # print(len(file_names))
  # for i in range(0, len(file_names)): 
  # for ele in file_names:
  #   print(i)
  #   res = p.match(file_names[i])
  #   if(res == None):
  #     file_names.pop(i)
  # print(file_names)





  ## Read 1번째 사진
  printComment("read Image")
  # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
  image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[0]))


  ## Run detection
  results = model.detect([image], verbose=1)
  printComment("Detect Completion")
  r = results[0]


  ## Visualize results (결과값을 저장하고 사진으로 띄워서 보여준다.)
  # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


  ## Delete masks element
  printComment("_DELETE mask")
  del r['masks']
  printComment("results : ")
  print(r)


  ## Erase Picture
  command = "rm " + IMAGE_DIR + "/*"
  print(command)
  printComment("Erase Image DIR")
  os.system(command)
  time.sleep(0.1)  # 0.1초






