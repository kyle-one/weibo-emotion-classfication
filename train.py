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
glove_path="/Public/YongkunLiu/2021/weibo-course-design/dataset/sgns.weibo.word"

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(message)s")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    if not os.path.exists(args.WORK_DIR+'/log'):
        os.makedirs(args.WORK_DIR+'/log')
        #LOG_DIR +'/'+ RNN_TYPE+'_'+BIDIRECTIONAL+'_split'+str(SPLIT)+'.log'
    
    fHandler = logging.FileHandler("{}/{}.log".format(args.WORK_DIR+'/log',localtime), mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    return logger

def get_train_val_dataset_df(path):
    #读取数据集 1表示积极，0表示消极
    file = pd.read_csv(path)
    df = pd.DataFrame(file)
    dataset_names = ['train', 'valid']
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
    #StratifiedShuffleSplit 数据集划分函数
    #n_splits是train/test的组数，test_size为测试集的比例，andom_state控制是将样本随机打乱
    train_split_idx, val_split_idx = next(iter(stratified_split.split(df.review, df.label)))
    train_df = df.iloc[train_split_idx].reset_index()
    val_df = df.iloc[val_split_idx].reset_index()
    return train_df,val_df

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
    

def evaluate_loss(data_iter, net,hidden,device, epoch):
    net.eval()
    l_sum, n,val_pred_sum= 0.0, 0,0
    with torch.no_grad():
        for bidx,(X, y) in enumerate(data_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            y=y.to(device)
            X=X.to(device)
            
            y_hat,hidden= net(X,hidden)
            loss = criterion(y_hat, y)
            
            l_sum+=loss
            #l_sum +=criterion(y_hat, y).item()
            n += X.size()[1]
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            val_pred_sum+=pred_sum
    print_json={"epoch":str(epoch),"trainval":"val","loss":str(float(l_sum/(bidx+1))),"acc":val_pred_sum/n}
    logger.info(print_json)
    #print('val,epoch:{},pred:{},loss:{}'.format(epoch,val_pred_sum/n,l_sum/n))
    # logger.info("[epoch {}][{}][end] val_loss={:.5f},val_acc:{:.5f}({}/{})".format\
    # (epoch,'val',l_sum/(bidx+1),val_pred_sum/n,int(val_pred_sum),n))
    return l_sum / (bidx+1),val_pred_sum/n

def train(net,train_iter,val_iter,num_epoch,lr_period,lr_decay):
    optimizer = torch.optim.Adam(net.parameters(),lr=args.LR)
    hidden=None
    Max_Acc=0.0
    Min_loss=9999999.9
    for epoch in range(num_epoch):
        net.train()
        n,train_l_sum,train_pred_sum=0,0,0
        if epoch > 0 and epoch % lr_period == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*lr_decay
        for bidx,(X,Y) in enumerate(train_iter):
            # print(bidx)
            X=X.float()
            X=X.permute(1,0,2)
            X=X.to(ctx)
            Y=Y.to(ctx)
            if(hidden is not None):
                if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                    hidden[0].to(ctx)
                    hidden[1].to(ctx)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                else:   
                    hidden.to(ctx)
                    hidden = hidden.detach()
            
            optimizer.zero_grad()
            #print(str(bidx)+'---------------')
            #print(hidden)
            # print(X,hidden)
            y_hat,hidden= net(X,hidden)
            #print(y_hat.size())
            loss = criterion(y_hat, Y)
            #print(y_hat)
            #print(y)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            #print(loss)
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            # print(pred)
            pred_sum=pred.eq(Y.view_as(pred)).sum().item()
            #print(pred_sum)
            train_l_sum+=loss
            train_pred_sum+=pred_sum
            n+=X.size(1)
            print_json={"epoch":str(epoch),"batch":str(bidx),"trainval":"train","loss":str(float(loss)),"acc":pred_sum/Y.size()[0]}
            logger.info(print_json)

        if not os.path.exists(args.WORK_DIR+'params'):
            # print(args.WORK_DIR+'params')
            os.makedirs(args.WORK_DIR+'params')
            #{}
        print_json={"epoch":str(epoch),"batch":"end","trainval":"train","loss":str(float(train_l_sum/(bidx+1))),"acc":train_pred_sum/n}
        logger.info(print_json)
        valid_loss,valid_acc=evaluate_loss(val_iter, net,hidden,ctx,epoch)
        if(valid_acc>Max_Acc):
            Max_Acc=valid_acc
            model_best=net
            print_json={"epoch":str(epoch),"save":"save"}
            logger.info(print_json)
            torch.save(net,args.WORK_DIR+'params/best_{}.pth'.format(localtime))
        # logger.info("[epoch {}][{}][end] train_loss={:.5f},loss_classfication={:.5f},loss_confidence={:.5f},train_acc={:.5f}({}/{})".format\
        #             (epoch,'train',train_l_sum/(bidx+1),loss_classfication_sum/(bidx+1),loss_confidence_sum/(bidx+1),train_pred_sum/n,int(train_pred_sum),n))
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WILDCAT Training')
    parser.add_argument("--GLOVE_PATH", default="/Public/YongkunLiu/2021/weibo-course-design/dataset/sgns.weibo.word",help="display a square of a given number", type=str)
    parser.add_argument("--NUM_EPOCHS",default=10, help="display a square of a given number", type=int)
    parser.add_argument("--BATCH_SIZE",default=256, help="display a square of a given number", type=int)
    parser.add_argument("--IMG_SIZE",default=512, help="display a square of a given number", type=int)
    parser.add_argument("--LR",default=0.01, help="display a square of a given number", type=float)
    parser.add_argument("--WORK_DIR",default='/Public/YongkunLiu/weibo-workdir/', help="display a square of a given number", type=str)
    parser.add_argument("--TRAINVAL_PATH",default="./dataset/weibo_senti_100k_trainval_fenci.csv", help="display a square of a given number", type=str)
    parser.add_argument("--HIDDEN_SIZE",default=32, help="display a square of a given number", type=int)
    parser.add_argument("--NUM_LAYERS",default=2, help="display a square of a given number", type=int)
    parser.add_argument("--BIDIRECTIONAL",default="True", help="display a square of a given number", type=str)
    parser.add_argument("--RNN_TYPE",default="GRU", help="display a square of a given number", type=str)
    
    args = parser.parse_args()

    ctx=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    localtime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
    
    logger=get_logger()
    print_json={"time":str(localtime),"epoch":args.NUM_EPOCHS,"BATCH_SIZE":args.BATCH_SIZE,
                "RNN_TYPE":args.RNN_TYPE,"BIDIRECTIONAL":args.BIDIRECTIONAL,"NUM_LAYERS":args.NUM_LAYERS,
               "HIDDEN_SIZE":args.HIDDEN_SIZE}
    logger.info(print_json)
    
    train_df,val_df=get_train_val_dataset_df(args.TRAINVAL_PATH)
    train_dataset=Dataset(train_df,100)
    val_dataset=Dataset(val_df,100)
    train_iter =DataLoader(train_dataset,batch_size=args.BATCH_SIZE,shuffle=False,num_workers=1,drop_last=True)
    val_iter =DataLoader(val_dataset,batch_size=args.BATCH_SIZE,shuffle=False,num_workers=1,drop_last=True)
    
    
    with open("./dataset/glove.json",'r',encoding='utf-8') as f:
        glove_dict=json.load(f)

    model=RNN(hidden_size=args.HIDDEN_SIZE,num_layers=args.NUM_LAYERS,bidirectional=args.BIDIRECTIONAL,rnn_type=args.RNN_TYPE)
    model.to(ctx)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    train(model,train_iter,val_iter,args.NUM_EPOCHS,2,0.5)
    

    