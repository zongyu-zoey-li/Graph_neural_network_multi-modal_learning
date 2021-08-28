import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import scipy.io
import torch
import dgl
import torch.nn as nn
import torch
import scipy.io
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from dgl.nn import GraphConv

''' Graph mining final project: Zongyu Li 
the embedding (newKinematics2.mat) was provided by Shohaib Mahmud 
'''

dataDir = '/home/aurora/Documents/final_graph_mining/newKinematics2.mat'
test_size = 0.1

src_ids = [0,0,0,0,1,1,1,2,2,3,1,2,3,4,2,3,4,3,4,4]

dst_ids = [1,2,3,4,2,3,4,3,4,4,0,0,0,0,1,1,1,2,2,3]
g = dgl.graph((src_ids, dst_ids))

def dataLoader(dataDir, test_size):
    data = scipy.io.loadmat(dataDir)['all']
    data_train, data_test = train_test_split(data, test_size=test_size)
    return data_train, data_test

class TimeseriesNet_light(nn.Module):
    def __init__(self):
        super(TimeseriesNet_light,self).__init__()
        # self.festures=13
        self.seq_len =60
        # self.hidden_dim = 1024
        # self.layer_dim =1
        self.stage_1_conv_x=nn.Conv1d(3,64,kernel_size=5,stride=2)
        self.stage_1_pool_x = nn.MaxPool1d(2,2)
        self.stage_1_drop_x = nn.Dropout(p=0.2)
        self.stage_1_norm_x = nn.BatchNorm1d(64)
        self.stage_2_conv_x = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool_x = nn.MaxPool1d(2,2)
        self.stage_2_drop_x = nn.Dropout(p=0.2)
        self.stage_2_norm_x = nn.BatchNorm1d(128)
        self.linear1_x = nn.Linear(256,168)
        self.linear2_x = nn.Linear(168,60)
        self.linear3_x = nn.Linear(60,32)
        
        
        
        self.stage_1_conv_x2=nn.Conv1d(3,64,kernel_size=5,stride=2)
        self.stage_1_pool_x2 = nn.MaxPool1d(2,2)
        self.stage_1_drop_x2 = nn.Dropout(p=0.2)
        self.stage_1_norm_x2 = nn.BatchNorm1d(64)
        self.stage_2_conv_x2 = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool_x2 = nn.MaxPool1d(2,2)
        self.stage_2_drop_x2 = nn.Dropout(p=0.2)
        self.stage_2_norm_x2 = nn.BatchNorm1d(128)
        self.linear1_x2 = nn.Linear(256,168)
        self.linear2_x2 = nn.Linear(168,60)
        self.linear3_x2 = nn.Linear(60,32)
        
        
        self.stage_1_conv = nn.Conv1d(10,64,kernel_size=5,stride=2)
        self.stage_1_pool = nn.MaxPool1d(2,2)
        self.stage_1_drop = nn.Dropout(p=0.2)
        self.stage_1_norm = nn.BatchNorm1d(64)
        self.stage_2_conv = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool = nn.MaxPool1d(2,2)
        self.stage_2_drop = nn.Dropout(p=0.2)
        self.stage_2_norm = nn.BatchNorm1d(128)
        
        
        self.stage_1_conv_2 = nn.Conv1d(10,64,kernel_size=5,stride=2)
        self.stage_1_pool_2 = nn.MaxPool1d(2,2)
        self.stage_1_drop_2 = nn.Dropout(p=0.2)
        self.stage_1_norm_2 = nn.BatchNorm1d(64)
        self.stage_2_conv_2 = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool_2 = nn.MaxPool1d(2,2)
        self.stage_2_drop_2 = nn.Dropout(p=0.2)
        self.stage_2_norm_2 = nn.BatchNorm1d(128)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(256,168)
        # self.linear2 = nn.Linear(336,168)
        self.linear2 = nn.Linear(168,60)
        self.linear3 = nn.Linear(60,32)
        
        self.linear1_2 = nn.Linear(256,168)
        # self.linear2 = nn.Linear(336,168)
        self.linear2_2 = nn.Linear(168,60)
        self.linear3_2 = nn.Linear(60,32)
        
        
        
        self.graph_1=GraphConv(32,32)
        self.graph_2=GraphConv(32,32)
        ## linear for prediction evaluation 
        self.linear_p1 = nn.Linear(160,90)
        self.linear_p2 = nn.Linear(90,1)
        # self.linear_p3 = nn.Linear(l2,1)
        self.initialize_weights()
        
        #barch normalization
        #self.stage_2_conv = nn.Conv1d()
    def forward(self,l,lx,r,rx,z,g):

        
        l = F.relu(self.stage_1_conv(l))
        l = self.stage_1_pool(l)
        l = self.stage_1_drop(l)
        l = self.stage_1_norm(l)
        l = F.relu(self.stage_2_conv(l))
        l = self.stage_2_pool(l)
        l = self.stage_2_drop(l)
        l = self.stage_2_norm(l)
        l = self.flat(l)
        l = F.relu(self.linear1(l))
        l = F.relu(self.linear2(l))
        l = F.relu(self.linear3(l))
        
        lx = F.relu(self.stage_1_conv_x(lx))
        lx = self.stage_1_pool_x(lx)
        lx = self.stage_1_drop_x(lx)
        lx = self.stage_1_norm_x(lx)
        lx = F.relu(self.stage_2_conv_x(lx))
        lx = self.stage_2_pool_x(lx)
        lx = self.stage_2_drop_x(lx)
        lx = self.stage_2_norm_x(lx)
        lx = self.flat(lx)
        lx = F.relu(self.linear1_x(lx))
        lx = F.relu(self.linear2_x(lx))
        lx = F.relu(self.linear3_x(lx))
  
          
        r = F.relu(self.stage_1_conv_2(r))
        r = self.stage_1_pool_2(r)
        r = self.stage_1_drop_2(r)
        r = self.stage_1_norm_2(r)
        r = F.relu(self.stage_2_conv_2(r))
        r = self.stage_2_pool_2(r)
        r = self.stage_2_drop_2(r)
        r = self.stage_2_norm_2(r)
        r = self.flat(r)
        r = F.relu(self.linear1_2(r))
        r = F.relu(self.linear2_2(r))
        r = F.relu(self.linear3_2(r))
        
        rx = F.relu(self.stage_1_conv_x2(rx))
        rx = self.stage_1_pool_x2(rx)
        rx = self.stage_1_drop_x2(rx)
        rx = self.stage_1_norm_x2(rx)
        rx = F.relu(self.stage_2_conv_x2(rx))
        rx = self.stage_2_pool_x2(rx)
        rx = self.stage_2_drop_x2(rx)
        rx = self.stage_2_norm_x2(rx)
        rx = self.flat(rx)
        rx = F.relu(self.linear1_x2(rx))
        rx = F.relu(self.linear2_x2(rx))
        rx = F.relu(self.linear3_x2(rx))
        
        comb = torch.cat((l,lx,r,rx,z.squeeze().unsqueeze(-1).T),0)   
        
        gl = self.graph_1(g,comb.to(torch.float32))
        gl = F.relu(gl)
        gl = F.relu(self.graph_2(g,gl))
        
        gl=self.flat(gl)
        
        val = F.relu(self.linear_p1(gl.view(-1)))
        val = F.relu(self.linear_p2(val))
        val = torch.sigmoid(val)
        
        return val
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)






data_train, data_test = dataLoader(dataDir, test_size)

# For k-fold cross validation based training
numOfFolds = 10
numOfEpochs = 10
kf = KFold(n_splits=numOfFolds)
val_acc = 0
val_prec = 0
val_recall = 0

i=0;

for train_index, val_index in kf.split(data_train):
    # model definition here
    model=TimeseriesNet_light()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.02, weight_decay=0.0005)
    criterion = nn.BCELoss()
    i=i+1
    if i!=1: continue
    # training loop
    epo=0
    for n in range(numOfEpochs):
        epo=epo+1
        shuffledTrainingData = np.copy(data_train[train_index])
        np.random.shuffle(shuffledTrainingData)
        model.train()
        print(n)


        for data in shuffledTrainingData:
            data_=torch.from_numpy(data[0])
            mean_f=torch.mean(data_,0)
            std_f=torch.std(data_,0)
            data_=(data_-mean_f)/std_f
            a1=torch.tensor(np.zeros((1,10,60)),dtype=torch.float32)
            a2=torch.tensor(np.zeros((1,10,60)),dtype=torch.float32)
            b1=torch.tensor(np.zeros((1,3,60)),dtype=torch.float32)
            b2=torch.tensor(np.zeros((1,3,60)),dtype=torch.float32)
            Lx = data_[:, 0:3].T
            L = data_[:, 3:13].T
            Rx = data_[:, 13:16].T
            R = data_[:, 16:26].T
            a1[0,:,:]=L
            L= a1
            a2[0,:,:]=R
            R=a2
            b1[0,:,:]=Lx
            Lx=b1
            b2[0,:,:]=Rx
            Rx=b2
            
            optimizer.zero_grad()
            embedd = torch.from_numpy(data[3]).unsqueeze(-1)
            label = torch.tensor(int(data[2]=='nor'))
            y_ = torch.squeeze(model(L,Lx,R,Rx,embedd,g))
            loss = criterion(y_.view(-1).to(dtype=float),label.view(-1).to(dtype=float))
            loss = torch.autograd.Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()
            # forward pass
            # loss calc
            # backward pass

        # validation check
        
        
        correct = 0
        total = 0
        TP=0
        FP=0
        FN=0
        with torch.no_grad():
            model.eval()
            for data in data_train[val_index]:
                data_=torch.from_numpy(data[0])
                mean_f=torch.mean(data_,0)
                std_f=torch.std(data_,0)
                data_=(data_-mean_f)/std_f
                a1=torch.tensor(np.zeros((1,10,60)),dtype=torch.float32)
                a2=torch.tensor(np.zeros((1,10,60)),dtype=torch.float32)
                b1=torch.tensor(np.zeros((1,3,60)),dtype=torch.float32)
                b2=torch.tensor(np.zeros((1,3,60)),dtype=torch.float32)
                Lx = data_[:, 0:3].T
                L = data_[:, 3:13].T
                Rx = data_[:, 13:16].T
                R = data_[:, 16:26].T
                a1[0,:,:]=L
                L= a1
                a2[0,:,:]=R
                R=a2
                b1[0,:,:]=Lx
                Lx=b1
                b2[0,:,:]=Rx
                Rx=b2
                embedd = torch.from_numpy(data[3]).unsqueeze(-1)
                label =  torch.tensor(int(data[2]=='nor'))
                outputs= torch.squeeze(model(L,Lx,R,Rx,embedd,g))
                total +=outputs.cpu().numpy().size
                outputs_val=outputs>0.5
                correct+=(outputs_val == label.view(-1)).sum().item()
                TP+=torch.sum(outputs_val == label.view(-1)*label.view(-1)).item()
                FP+=torch.sum(outputs_val != label.view(-1)*outputs_val).item()
                FN+=torch.sum(outputs_val != label.view(-1)*label.view(-1)).item()
            
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            accuracy=correct/total
            F1=2*precision*recall/(precision+recall)
            print('interation :{}, f1:{}, precison:{}, recall:{}'.format(epo,F1,precision,recall))
                    
                
                # forward pass
                # accuracy, precision, recall calculation

        # network.train()
        # break

# avg_val_acc, avg_val_prec, avg_val_recall

correct = 0
total = 0
TP=0
FP=0
FN=0
# test the best model on the test data set
with torch.no_grad():
    model.eval()
    for data in data_test:
        # test_accuracy, test_precision, test_recall
        data_=torch.from_numpy(data[0])
        mean_f=torch.mean(data_,0)
        std_f=torch.std(data_,0)
        data_=(data_-mean_f)/std_f
        a1=torch.tensor(np.zeros((1,10,60)),dtype=torch.float32)
        a2=torch.tensor(np.zeros((1,10,60)),dtype=torch.float32)
        b1=torch.tensor(np.zeros((1,3,60)),dtype=torch.float32)
        b2=torch.tensor(np.zeros((1,3,60)),dtype=torch.float32)
        Lx = data_[:, 0:3].T
        L = data_[:, 3:13].T
        Rx = data_[:, 13:16].T
        R = data_[:, 16:26].T
        a1[0,:,:]=L
        L= a1
        a2[0,:,:]=R
        R=a2
        b1[0,:,:]=Lx
        Lx=b1
        b2[0,:,:]=Rx
        Rx=b2
        embedd = torch.from_numpy(data[3]).unsqueeze(-1)
        label =  torch.tensor(int(data[2]=='nor'))
        outputs= torch.squeeze(model(L,Lx,R,Rx,embedd,g))
        total +=outputs.cpu().numpy().size
        outputs_val=outputs>0.5
        correct+=(outputs_val == label.view(-1)).sum().item()
        TP+=torch.sum(outputs_val == label.view(-1)*label.view(-1)).item()
        FP+=torch.sum(outputs_val != label.view(-1)*outputs_val).item()
        FN+=torch.sum(outputs_val != label.view(-1)*label.view(-1)).item()
            
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    accuracy=correct/total
    F1=2*precision*recall/(precision+recall)
    print('test performance, f1:{}, precison:{}, recall:{}'.format(F1,precision,recall))
        