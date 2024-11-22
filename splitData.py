import os
import shutil
import random   
from itertools import islice

outputFolderPath = 'Dataset/SplitData'
inputFolderPath = 'Dataset/All'
splitRatio = {"train":0.7,"val":0.2,"test":0.1}
classes =["fake","real"]
try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    os.mkdir(outputFolderPath)
    
#   ------- Directory for Create -------
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

#   ------- Get the Names -------
listNames = os.listdir(inputFolderPath)
# print(listNames)
# print(len(listNames))
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))
# print(set(uniqueNames))
# print(len(uniqueNames))

#   ------- Shuffle -------
random.shuffle(uniqueNames)

#   ------- Find the number images for each folder -------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio["train"])
lenVal = int(lenData * splitRatio["val"])
lenTest = int(lenData * splitRatio["test"])
# print(f'Total Images: {lenData}\n Split: {lenTrain} {lenVal} {lenTest}')

#   ------- Put remaining images in Training-------
if lenData != lenTrain + lenVal + lenTest:
    remaining = lenData - (lenTrain + lenVal + lenTest)
    lenTrain += remaining


#   ------- Slip the list-------
lengthToSplit = [lenTrain, lenVal, lenTest]
input = iter(uniqueNames)
output = [list(islice(input, elem)) for elem in lengthToSplit]
print(f'Total Images: {lenData}\n Split: {len(output[0])} {len(output[1])} {len(output[2])}')
#   ------- Copy the files -------
sequence = ["train","val","test"]
for i,out in enumerate(output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg',f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt',f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Done")
#   ------- Create data.yaml -------
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'
        
f = open(f'{outputFolderPath}/data.yaml','a')
f.write(dataYaml)
f.close()

print("Data.yaml Done")

#   ------- Save the names -------

