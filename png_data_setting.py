import os
import cv2
from numpy import where
from matplotlib import pyplot as plt
import numpy as np 
import time
import glob
import pandas as pd
from tqdm import tqdm
import random
import pickle



def cutImg(img):
    #mask範囲してい
    # マスク範囲を四角形で描画
    boxFromX = 640 #マスク範囲開始位置 X座標
    boxFromY = 500 #マスク範囲開始位置 Y座標
    boxToX = 1280 #マスク範囲終了位置 X座標
    boxToY = 760 #マスク範囲終了位置 Y座標
    cut_img = img[400:550, 640:1280]
    return cut_img



files = glob.glob('C:\\Users\\rikua\\Documents\\screen_shot2\\*')

def filenameToInt(filename):
    name_list = filename.split("_")
    sum = int(name_list[0]) * 1000000 + int(name_list[1]) * 10000 + int(name_list[2]) * 100 + int(name_list[3])
    return sum



def save_png():
    for file in files:
        
        filename = file.split("/")[4] #.png入ってる
        filename, ext = os.path.splittext(filename) # 入ってない
        for i in range(len(df)): #すべての行
            begin = filenameToInt(df["bigintime"][i]) #最初と最後
            end = filenameToInt(df["endtime"][i])
            filevalue = filenameToInt(filename) 
            if filevalue > begin and filevalue < end: #最初と最後にあるときに
                img = cv2.imread(file) #read
                cut_img = cutImg(img) #cut
                new_filename = filename + str(df["visib"][i]) #file名をvisibアリにする
                cv2.imwrite(path_write + new_filename, cut_img)
                break

def create_df():
    train_data = []
    df = pd.read_excel("C:/Users/rikua/Documents/visib_data.xlsx")
    print(df)
    for i in tqdm(range(len(files))):
        file = files[i]
        # filename = file.split("/")[6] #.png入ってる
        # filename, ext = os.path.splitext(filename) # 入ってない
        # print(filename)
        filename = os.path.basename(file).split('.', 1)[0]
        # filename = file.split("\")[5] #.png入ってる
        # filename = filename.split(".")[0] #入ってない
        for i in range(len(df)): #すべての行
            begin = filenameToInt(df["bigintime"][i])
            end = filenameToInt(df["endtime"][i])
            filevalue = filenameToInt(filename)
            if filevalue > begin and filevalue < end :
                img = cv2.imread(file)
                cut_img = cutImg(img)
                # cv2.imshow("Image", cut_img)
                # cv2.waitKey()
                train_data.append([cut_img, df["visib"][i]])
                break
    train_df = pd.DataFrame(train_data)
    print(train_df)
    random.shuffle(train_data)
    x_train = []
    y_train = []
    for feature, label in train_data:
        x_train.append(feature)
        y_train.append(label)
    X_train = np.array(x_train)
    Y_train = np.array(y_train)
    #Data = np.array(train_data)
    with open('X_train.pickle', mode='wb') as fo:
        pickle.dump(X_train, fo)
    with open('Y_train.pickle', mode='wb') as fo:
        pickle.dump(Y_train, fo)     
    # with open('train_data.pickle', mode='wb') as fo:
    #     pickle.dump(Data, fo)    

if __name__ == "__main__":
    create_df()

