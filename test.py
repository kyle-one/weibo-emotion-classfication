import os
import torch
import numpy as np
import jiagu
import jieba
import torchtext
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import logging
import time
import json

class RNN(nn.Module):
    def __init__(self, input_size=300, hidden_size=32, output_size=2,num_layers=2,bidirectional='True',rnn_type='GRU'):#输入，隐藏层单元，输出，隐藏层数，双向
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        if(bidirectional=='True'):
            self.bidirectional=True
        elif(bidirectional=='False'):
            self.bidirectional=False
        if(rnn_type=='GRU'):
            self.rnn=nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='LSTM'):
            self.rnn=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='RNN'):
            self.rnn=nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        
        if(bidirectional=='True'):
            self.out = nn.Linear(4*hidden_size, output_size)
        else:
            self.out = nn.Linear(1*hidden_size, output_size)

    def forward(self, input, hidden):
        #print(input.size())
        #10*64*8
        output1,hidden=self.rnn(input,hidden)
        #print("output1:{}".format(output1.size()))
        #10*64*32
        if(self.bidirectional==True):
            output2 = torch.cat((output1[0], output1[-1]), -1)
        else:
            output2 = output1[-1]
        #print("output2:{}".format(output2.size()))
        #64*64
        #print("hidden:{}".format(hidden.size()))
        #层数*64*结点数
        #2*64*32
        output=self.out(output2)
        #print(output.size())
        #64*4
        #output=F.softmax(output,dim=0)
        return output, hidden
#将句子分词，并且转化为词向量的模块，maxlen为一个句子的长度
#要不要提前分好词，算了...嫌麻烦，先用着
#有了这个模块，似乎就不需要词向量嵌入层nn.Embedding层了？
def text_pipeline(text,maxlen):
    # text_list=jieba.cut(text, cut_all=False)
    # text_list=list(text_list)
    # print(text_list)
    #将glove字典中找不到的词语分为若干个字
    text_list=json.loads(text)
    # print(text_list)
    while 1:
        flag=0
        for idx,i in enumerate(text_list):
            #分为最小，若最小也在glove中无，则删除
            if i not in glove_dict:
                # print(i)
                text_list.remove(i)
                for j in i[::-1]:
                    if j in glove_dict:
                        text_list.insert(idx,j)
                flag=1
                break
        if flag==0:
            break
            
    d=maxlen-len(text_list)
    # print(text_list)
    if d<=0:
        text_list=text_list[:maxlen]
    # else:
    #     for i in range(d):
    #         text_list.append("。")
            
    #构建array，将glove词向量拼接成为(100,300)形状
    res=np.zeros((maxlen, 300))
    # print(text_list)
    for idx,i in enumerate(text_list):
        res[idx]=np.array(glove_dict[i])
    
    # text_list=[torch.tensor(glove_dict[i], dtype=torch.float64) for i in text_list]#torch.tensor(glove_dict[i], dtype=torch.float64)
    res=torch.tensor(res,dtype=torch.float32)
    
    return res

class Dataset(Dataset):
    def __init__(self,labels_df,maxlen,transform=None):
        super().__init__()
        self.labels_df = labels_df
        self.transform = transform
        self.maxlen=maxlen

    def __len__(self):
        return self.labels_df.shape[0]


    def __getitem__(self, idx):
        review=self.labels_df.review[idx]
        review=text_pipeline(review,self.maxlen)
        label=self.labels_df.label[idx]
        return review,label
        
        
        # image_name = config.PATH_IMG+'/'+self.labels_df.id[idx]+'.jpg'
        # img = Image.open(image_name)
        # label = self.labels_df.label_idx[idx]
        # if self.transform:
        #     img = self.transform(img)
        # return img, label

def test(model_path):
    model=torch.load(model_path)
    sum=0
    test_len=0
    for i in test_dataset:
        test_len+=1
        if test_len>=test_dataset.__len__():
            break
        content = i[0].unsqueeze(0)
        content = content.to(ctx)
        content = content.permute(1, 0, 2)
        pred = model(content, None)[0]
        pred = pred.max(1, keepdim=True)[1].view(-1).cpu().numpy().tolist()
        if pred[0]==i[1]:
            sum+=1
    return sum/11999

if __name__ == '__main__':
    log_path="/Public/YongkunLiu/weibo-workdir/log"
    params_path="/Public/YongkunLiu/weibo-workdir/params"
    ctx=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_list=os.listdir(log_path)
    with open("/Public/YongkunLiu/2021/weibo-course-design/dataset/glove.json",'r',encoding='utf-8') as f:
        glove_dict=json.load(f)
    file = pd.read_csv("/Public/YongkunLiu/2021/weibo-course-design/dataset/weibo_senti_100k_test_fenci.csv")
    test_df = pd.DataFrame(file)
    test_dataset=Dataset(test_df,100)
    res_list=[]
    for file_name in log_list:
        if "log" in file_name:
            with open(log_path+"/"+file_name) as f:
                a=f.readlines()
                info=eval(a[0])
                # print(info)
                # print(file_name)
                model_path=params_path+"/best_"+file_name.split(".log")[0]+".pth"
                info["acc"]=test(model_path)
                res_list.append(info)
                print(info)