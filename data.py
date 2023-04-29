import os
import shutil
import random

#OPTIONS



def getFiles(path:str) -> list:
    return os.listdir(path)


if "newData" not in getFiles("."):
    os.mkdir("newData")
else:
    os.chdir("newData")
    for f in getFiles("."):
        os.remove(f)
    os.chdir("..")





os.chdir("dataset")


listOfImages = []

for folder in getFiles("."):
    os.chdir(folder)
    for index,file in enumerate(getFiles(".")):
        #os.rename(file, f"{folder}-{index}.png")
        listOfImages.append(f"dataset\\{folder}\\{folder}-{index}.png")
    os.chdir("..")
os.chdir("..")
print(listOfImages)
index = 0
for path in listOfImages:
    
    shutil.copy2(os.getcwd()+"\\"+path,os.getcwd()+"\\"+"newData") 
    
    