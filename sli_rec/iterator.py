import numpy as np
import json
import pickle as pkl
import random
import math
from utils import shuffle

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            f_json = json.load(f) #load file json
            return dict((key.encode("UTF-8"), value) for (key,value) in f_json.items())
    except:
        with open(filename, 'rb') as f:
            f_pkl = pkl.load(f) #load file pickle
            return dict((key.encode("UTF-8"), value) for (key,value) in f_pkl.items())
    

class Iterator: #dung de iterate data in batches
    def __init__(self, source,
                 uid_voc="data/user_vocab.pkl",
                 mid_voc="data/item_vocab.pkl",
                 cat_voc="data/category_vocab.pkl",
                 batch_size=128,
                 max_batch_size=20):
# source: The data source to iterate over (train and test data)
# uid_voc: The filename for the user vocabulary pickle file.
# mid_voc: The filename for the item vocabulary pickle file.
# cat_voc: The filename for the category vocabulary pickle file.
# batch_size: The desired batch size for the iteration.
        self.source0 = source #store original data
        self.source = shuffle(self.source0) #shuffle r luu vao file .shuff
        self.userdict,  self.itemdict, self.catedict = load_dict(uid_voc), load_dict(mid_voc), load_dict(cat_voc)
        self.batch_size = batch_size
        self.k = batch_size * max_batch_size #maximum number of elements
        self.end_of_data = False
        self.source_buffer = [] #buffer to store data batches

    def __iter__(self): 
        return self #khi su dung for loop de iter qua iterable object thi ham nay se dc goi. No tra ve self tuc la no tra ve chinh cai iterable object

    def reset(self):
        self.source= shuffle(self.source0)
        
    def get_id_numbers(self):
        print("GET_ID_NUMBERS\n\n\n\n\n")
        return len(self.userdict), len(self.itemdict), len(self.catedict) #total data

    def __next__(self):
        
        if self.end_of_data: #train het r thi reset lai tu dau
            self.end_of_data = False
            self.reset()
            raise StopIteration
# --------------------------------------------
        source = [] #x
        target = [] #y
        if len(self.source_buffer) == 0: #check xem co j trong source_buffer k
            # print(self.userdict)
            for k_ in range(self.k):
                
                ss = self.source.readline() #lay 1 dong trong self.source (shuffled train file)
                # print(ss)
                if ss == b"": #check xem da den cuoi file chua
                    break
                self.source_buffer.append(ss.strip(b"\n").split(b"\t")) #lay thong tin trong file source r ghi vao file buffer. Bay gio file self.source_buffer se chua data cua file train 

            his_length = np.array([len(s[5].decode('utf-8').split("")) for s in self.source_buffer]) #s[5]: movie id list. his_length la 1 mang va se luu so luong movie o moi dong cua file self.source_buffer
            tidx = his_length.argsort()
            _sbuf = [self.source_buffer[i] for i in tidx]
            self.source_buffer = _sbuf
            # sap xep self.source_buffer dua theo so luong cua movie, theo thu tu ascending

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            while True:
                try:
                    ss = self.source_buffer.pop() #remove last element
                except IndexError:
                    break

                uid = self.userdict[ss[1]] if ss[1] in self.userdict else 0 #Lay index cua userid dua vao userid
                mid = self.itemdict[ss[2]] if ss[2] in self.itemdict else 0 
                cat = self.catedict[ss[3]] if ss[3] in self.catedict else 0
                timestepnow = float(ss[4]) #unixtime
                
                tmp = []
                for fea in ss[5].decode('utf-8').split(""): #fea: movie id
                    m = self.itemdict[fea] if fea in self.itemdict else 0
                    tmp.append(m) #ghi index cua movie id vao tmp
                mid_list = tmp #movie id list bay gio se chua index cua cac movie, sap xep theo thu tu la so lan xuat hien cua movie

                tmp1 = [] #tuong tu buoc tren nhung lam voi category
                for fea in ss[6].decode('utf-8').split(""):
                    c = self.catedict[fea] if fea in self.catedict else 0
                    tmp1.append(c)
                cat_list = tmp1

                tmp2 = [] #tuong tu nhung lam voi unix time
                for fea in ss[7].decode('utf-8').split(""):
                    tmp2.append(float(fea))
                time_list = tmp2
                
                #Time-LSTM-123 
                tmp3 = []
                for i in range(len(time_list)-1): #so luong unix time cua dong ss
                    deltatime_last = (time_list[i+1] - time_list[i])/(3600 * 24) #so ngay giua 2 khoang tg
                    if deltatime_last <= 0.5: #de phong khoang thoi gian qua be
                        deltatime_last = 0.5
                    tmp3.append(math.log(deltatime_last))
                deltatime_now = (timestepnow - time_list[-1])/(3600 * 24) #stk = thoi diem hien tai - lan mua gan nhat
                if deltatime_now <= 0.5:
                    deltatime_now = 0.5    
                tmp3.append(math.log(deltatime_now))               
                timeinterval_list = tmp3

                #Time-LSTM-4
                tmp4 = []
                tmp4.append(0.0)
                for i in range(len(time_list)-1):
                    deltatime_last = (time_list[i+1] - time_list[i])/(3600 * 24)
                    if deltatime_last <= 0.5:
                        deltatime_last = 0.5
                    tmp4.append(math.log(deltatime_last))
                timelast_list = tmp4 #1 list luu khoang thoi gian giua nhung lan xem
                
                tmp5 = []
                for i in range(len(time_list)):
                    deltatime_now = (timestepnow - time_list[i])/(3600 * 24)
                    if deltatime_now <= 0.5:
                        deltatime_now = 0.5
                    tmp5.append(math.log(deltatime_now))
                timenow_list = tmp5 #1 list luu khoang thoi gian giua hien tai va cac lan xem trc

                source.append([uid, mid, cat, mid_list, cat_list, timeinterval_list, timelast_list, timenow_list])
                target.append([float(ss[0]), 1-float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target
    
