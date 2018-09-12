
# coding: utf-8
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
  print(C_BOLD + C_GREEN)
  print(str)
  print(C_END)
  # print(C_BOLD +  C_GREEN + str + C_END)
def printError(str):
  print(C_BOLD + C_RED)
  print(str)
  print(C_END)
  # print(C_BOLD +  C_RED + str + C_END)

from threading import Thread
from multiprocessing import Pool
from multiprocessing import Process
# from multiprocessing import Queue
import json
import http.client
import urllib
# import json
from collections import OrderedDict
import queue as Queue
import time
import re
import time
import threading
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
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco
## Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
## Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
## Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
## Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
printComment("Directory Print : ")
print("ROOT_DIR : " + ROOT_DIR)
print("IMAGE_DIR : " + IMAGE_DIR)
print("MODEL_DIR : " + MODEL_DIR)
print("COCO_MODEL_PATH : " + COCO_MODEL_PATH)


## Define JSON USER FUNCTION
printComment("Define JSON USER FUNCTION")
file_data = OrderedDict()
#JSON Data convert to Array data
def jsonToidx(json_list):
  index = 0
  arrayData = []
  for item in json_list:
      arrayData.insert(index,item)
      index += 1
  return arrayData
#Array Data convert to Name Data
def idxToName(id_list):
  index = 0
  arrayData = []
  for item in id_list:
      arrayData.insert(index,class_names[int(item)])
      index += 1
  return arrayData
#Send data to Server
def sendToServer(data_list):
  #Set Data

  params = urllib.parse.urlencode({'data':data_list})
  # b = params.encode('utf-8')
  # print(b)

  headers = {"Content-type": "application/x-www-form-urlencoded","Accept": "text/plain"}
  
  #Linking with Server, Update Host and port of server with Yuntae
  # http://kyt529.iptime.org:8080/CenterServer.jsp
  # conn = http.client.HTTPConnection("172.30.1.60",8080)
  # URL url = new URL("http://kyt529.iptime.org:8080/CenterServer.jsp")
  conn = http.client.HTTPConnection("kyt529.iptime.org",8080)
  # conn = http.client.HTTPConnection("172.20.10.6",8080)
  conn.connect()
  #Update directory server with Yuntae
  # conn.request("POST", "/JspServerTest/NewFile.jsp", params, headers)
  printComment(params)
  
  # conn.request("POST", "/JSPServer/NewFile.jsp", params, headers)
  conn.request("POST", "/var/lib/tomcat8/webapps/ROOT/index.jsp", params, headers)
  # conn.request("POST", "/server/CenterServer.jsp", params, headers)
  r1 = conn.getresponse()
  print(r1.status, r1.reason)
  conn.close() 
  print("SendToServer() Fnished!!!!")


## Return DIR Length
def search(dir):
        files = os.listdir(dir)
        # for file in files:
        #         print(file)
        return len(files)

# def search2(dir):
#         files = os.listdir(dir)
#         for file in files:
#                 fullFilename = os.path.join(dir, file)
#                 print(fullFilename)
# search2(IMAGE_DIR)


# def search3(dir):
#         files = os.listdir(dir)
#         for file in files:
#                 fullFilename = os.path.join(dir, file)
#                 if os.path.isdir(fullFilename):
#                         search(fullFilename)
#                 else:
#                         print(fullFilename)
# search3(IMAGE_DIR)


## Define JSON USER FUNCTION
printComment("Define Thread Pool Function")
# class Worker(Thread):
#     """Thread executing tasks from a given tasks queue"""
#     def __init__(self, tasks):
#         Thread.__init__(self)
#         self.tasks = tasks
#         self.daemon = True
#         self.start()

#     def run(self):
#         while True:
#             func, args, kargs = self.tasks.get()
#             try:
#                 func(*args, **kargs)
#             except Exception as e:
#                 print (e)
#             # finally:
#                 # self.tasks.task_done()

# class ThreadPool:
#     """Pool of threads consuming tasks from a queue"""
#     def __init__(self, num_threads):
#         self.tasks = Queue.Queue(num_threads)
#         for _ in range(num_threads):
#             Worker(self.tasks)

#     def add_task(self, func, *args, **kargs):
#         """Add a task to the queue"""
#         self.tasks.put((func, args, kargs))

#     def wait_completion(self):
#         """Wait for completion of all the tasks in the queue"""
#         res = self.tasks.join()
#         printComment("Join Return Compeletion")
#         return res

## Config COCO FUNC
printComment("Config COCO")
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

## Create model object in inference mode.
printComment("Create Model Object")
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

## Load weights trained on MS-COCO
printComment("LOAD MS-COCO")
model.load_weights(COCO_MODEL_PATH, by_name=True)

## COCO Class names
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


# def threadingJOB(imageArr):
#   printError ("Threading JOB")
#   for item in imageArr:
#     res = model.detect([item], verbose=0)
#     print(res)




# def poolMap(r1):
#   # try :
#   printComment("Detect Completion")
#   printComment("Config COCO")

#   global config
#   # MODEL_DIR = os.path.join(ROOT_DIR, "logs" + "/" + str(time.time()))
#   model1 = modellib.MaskRCNN(mode="inference", model_dir=os.path.join(ROOT_DIR, "logs" + "/" + str(time.time())), config=config)
#   # model1.load_weights(COCO_MODEL_PATH, by_name=True)
#   results = model1.detect([r1], verbose=1)  
#   print(results)
#   r = results[0]
#   printComment(r)
#   # except Exception as ex: # 에러 종류
#   #   printError("DETECT 에러 발생 ") # ex는 발생한 에러의 이름을 받아오는 변수
#   #   print(ex)
#   #   command = "rm " + IMAGE_DIR + "/" + file_names[0]
#   #   printComment(command)
#   #   printComment("Erase Image DIR")
#   #   os.system(command)


#   ## np.array to String
#   # example
  
#   # >>> x = np.array([1e-16,1,2,3])
#   # >>> print(np.array2string(x, precision=2, separator=',',
#   # ...                       suppress_small=True))
#   # [ 0., 1., 2., 3.]
  
#   printComment("JSON Data LOAD")
#   del r['masks']
#   # print(r)
  
#   ## class_ids
#   list1 = []
#   length = len(r['class_ids'])
#   for i in range(0, length):
#     list1.append(np.array2string(r['class_ids'][i], precision=2, separator=',', suppress_small=True))
#   r['class_ids'] = list1
#   # print(r['class_ids'])

#   ## rois
#   list2 = []
#   length = len(r['rois'])
#   for i in range(0, length):
#     length2 =  len(r['rois'][i])
#     list2_sub = []
#     for j in range(0, length2):
#       list2_sub.append(np.array2string(r['rois'][i][j], precision=2, separator=',', suppress_small=True))
#     list2.append(list2_sub)
#   r['rois'] = list2

#   ## scores
#   list3 = []
#   length = len(r['scores'])
#   for i in range(0, length):
#     list3.append(np.array2string(r['scores'][i], precision=2, separator=',', suppress_small=True))
#   r['scores'] = list3
#   # printComment(r)

#   ## dictionary --> String
#   r = json.dumps(r)
#   # printComment(r)
#   ## String --> JSON OBJECT
#   data = json.loads(r)

#   ## Devided Json Data
#   printComment("Devided JSON DATA")
#   pre_rois = data['rois']
#   pre_class_ids = data['class_ids']
#   pre_score = data['scores']
#   jsonidx = jsonToidx(pre_class_ids)
#   # print("JSON IDX : ", jsonidx)
#   idxName = idxToName(jsonidx)
#   # print("JSON idxName : ", idxName)

#   print(pre_rois)
#   print("----->jsonidx Data :",jsonidx)
#   print("----->idxName Data :",idxName)
#   print(pre_score)

#   ## Merge to Json Data
#   file_data = OrderedDict()
#   file_data["rois"] = pre_rois
#   file_data["class_names"] = idxName
#   file_data["scores"] = pre_score
#   print(file_data["rois"])
#   print(json.dumps(file_data,ensure_ascii=False,indent="\t"))
#   file_data = json.dumps(file_data,ensure_ascii=False,indent="\t")
#   ## Send File
#   printComment("Send to File")
#   sendToServer(file_data)

#   # Visualize results (결과값을 저장하고 사진으로 띄워서 보여준다.)
#   # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

#   ## Erase Picture
#   # command = "rm " + IMAGE_DIR + "/*"
#   # print(command)
#   # printComment("Erase Image DIR")
#   # os.system(command)
#   # listLength = 0

# pool = Pool(20)

## Erase Picture
command = "rm " + IMAGE_DIR + "/*"
print(command)
printComment("Erase Image DIR")
os.system(command)

## 변수 초기화
listLength = 0 # IMG_DIR안에 image 개수
jobList = [] # 처리할 이미지 담아두는 container



## 아두이노처럼 무한루프
while True :
  ## 추가 시 길이에 변화가 생기면서 루프문을 벗어난다.
  printComment("Length Checking...")
  while (search(IMAGE_DIR)) == 0:
   time.sleep(0.1)  # 0.1초
   printError("Time Sleep")
   continue

  ## Update Length
  printComment("Add New Image File")
  listLength = search(IMAGE_DIR) # IMAGE_DIR안에 이미지 개수 반환
  printComment("list Length : ")
  printComment(listLength)


  ## Excute Command 
  printComment("Remove /.DS_Store")
  # command = "rm " + IMAGE_DIR + "/*.json"
  # print(command)
  # os.system(command)
  command = "rm " + IMAGE_DIR + "/.DS_Store"
  printComment(command)
  os.system(command)


  ## Load first Image
  printComment("Load Image")
  # files = next(os.walk(IMAGE_DIR))
  # printComment(files)
  ## 파일 전체 리스트
  file_names = next(os.walk(IMAGE_DIR))[2]
  printComment( "Selected Image : ")
  print(file_names)
  print()


  ## file_names Filtering
  printComment("Filtering file_names")
  # p = re.compile('\.(jpg|gif|png)')
  p = re.compile('\w+\.(jpg|gif|png|jpeg)')
  file_names = list(filter(p.match, file_names)) # Read Note
  printComment(file_names)
  # sys.exit()
  print("Regex Completion")
  
  ## Read Image files
  printComment("Read Image")
  
  # jobList.extend(file_names)
  # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
  # for i in range(0, len(file_names)):
  #   image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
  #   jobList.append(image)

  image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


  printComment("JOB LIST PRINT : ")
  print(len(jobList))
  # print(jobList)
  # print(pool.map(poolMap, jobList))

  # p = Process(target=poolMap, args=(jobList[0],))
  # p.start()
  # p.join()


  # sleep(1)
  # if len(jobList) > 1:
  #   jobList.pop()

  ## Erase Picture
  # command = "rm " + IMAGE_DIR + "/*"
  # print(command)
  # printComment("Erase Image DIR")
  # os.system(command)
  # listLength = 0


  try :
    printComment("Detect Completion")
    # for i in range(0, len(jobList)):
    #   # print('처리 사진 : {}'.format(jobList[i]))
    #   results = model.detect([jobList[i]], verbose=1)  

    results = model.detect([image], verbose=0)
    # printComment("Return results")
    # results = model.detect(jobList, verbose=1)  
    r = results[0]

  except Exception as ex: # 에러 종류
    printError("DETECT 에러 발생 ") # ex는 발생한 에러의 이름을 받아오는 변수
    print(ex)
    command = "rm " + IMAGE_DIR + "/" + file_names[0]
    printComment(command)
    printComment("Erase Image DIR")
    os.system(command)
    continue


  ## np.array to String
  # example
  
  # >>> x = np.array([1e-16,1,2,3])
  # >>> print(np.array2string(x, precision=2, separator=',',
  # ...                       suppress_small=True))
  # [ 0., 1., 2., 3.]
  



  printComment("JSON Data LOAD")
  del r['masks']
  print(r)
  
  ## class_ids
  list1 = []
  length = len(r['class_ids'])
  for i in range(0, length):
    list1.append(np.array2string(r['class_ids'][i], precision=2, separator=',', suppress_small=True))
  r['class_ids'] = list1
  # print(r['class_ids'])

  ## rois
  list2 = []
  length = len(r['rois'])
  for i in range(0, length):
    length2 =  len(r['rois'][i])
    list2_sub = []
    for j in range(0, length2):
      list2_sub.append(np.array2string(r['rois'][i][j], precision=2, separator=',', suppress_small=True))
    list2.append(list2_sub)
  r['rois'] = list2

  ## scores
  list3 = []
  length = len(r['scores'])
  for i in range(0, length):
    list3.append(np.array2string(r['scores'][i], precision=2, separator=',', suppress_small=True))
  r['scores'] = list3






  # printComment(r)

  
  ## dictionary --> String





  r = json.dumps(r)
  printComment(r)
  ## String --> JSON OBJECT
  data = json.loads(r)
  ## Devided Json Data
  printComment("Devided JSON DATA")
  pre_rois = data['rois']
  pre_class_ids = data['class_ids']
  pre_score = data['scores']
  jsonidx = jsonToidx(pre_class_ids)
  # print("JSON IDX : ", jsonidx)
  idxName = idxToName(jsonidx)
  # print("JSON idxName : ", idxName)

  print(pre_rois)
  print("----->jsonidx Data :",jsonidx)
  print("----->idxName Data :",idxName)
  print(pre_score)

  ## Merge to Json Data
  file_data = OrderedDict()
  file_data["rois"] = pre_rois
  file_data["class_names"] = idxName
  file_data["scores"] = pre_score
  print("file_data : ",file_data)
  print()
  print(json.dumps(file_data,ensure_ascii=False,indent="\t"))
  file_data = json.dumps(file_data,ensure_ascii=False,indent="\t")
  print("json dumps file_data : ",file_data)

  ## Send File
  printComment("Send to File")
  printComment(file_data)
  sendToServer(file_data)
  



  ## Visualize results (결과값을 저장하고 사진으로 띄워서 보여준다.)
  # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
  # sys.exit()
  ## Delete masks element
  # printComment("_DELETE mask")
  # printComment("results : ")
  # print(r)


  ## Erase Picture
  command = "rm " + IMAGE_DIR + "/*"
  print(command)
  printComment("Erase Image DIR")
  os.system(command)
  listLength = 0





