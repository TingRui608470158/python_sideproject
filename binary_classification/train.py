#Fisrt do the classification task
#想先訓練一張圖有口罩或沒有口罩
import os 
import time
import numpy as np
import cv2
import argparse
import torch
import torchvision.models as models
import torch.nn.functional as fun
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable


if __name__ == '__main__':
    print("Training start")
    print("Torch version:",torch.__version__)
    ## cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/catdog", help="image path")
    args = parser.parse_args()
    
    for root, dirs, files in os.walk(".\data\catdog", topdown=False):
        print("root = ", root)
        print("dirs = ", dirs)
        # print("files = ", files)
        # for name in files:
        #     print(os.path.join(root, name))
        # for name in dirs:
        #     print(os.path.join(root, name))

    # ##模型
    # model = models.vgg16(pretrained=True)
    # model.classifier.add_module("7", nn.Linear(1000, 3))
    # model.classifier.add_module("8", nn.Softmax(dim = 1))
    # model = model.to(device)
 
    # print(model)
    # print(model._modules.keys())

    # ## 優化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # ## loss
    # criterion = nn.CrossEntropyLoss().cuda()

    # # 讀取資料集
    
    # batch_size = 1
    # dataloader = DataLoader(dataset=dataset, batch_size= batch_size, shuffle=True)
    # # dataiter = iter(dataloader)
    # # data = dataiter.next()
    # #訓練
    # model.train()
    # num_epochs = 3
    # for epoch in range(num_epochs):
    #     batch_size_start = time.time()
    #     running_loss = 0.0
    #     for i, (inputs, labels) in enumerate(dataloader):
    #         inputs = Variable(inputs.float()).cuda()
    #         labels = Variable(labels).cuda()
    #         # labels = Variable(fun.one_hot(labels, num_classes=3)).cuda()
    #         inputs = inputs.permute(0, 3, 2, 1)
    #         # inputs = torch.unsqueeze(inputs,dim = 0)
    #         # print(i,inputs.shape,"\n label = ",labels)
    #         optimizer.zero_grad()
            
    #         outputs = model((inputs).to(device))

    #         loss = criterion(outputs, labels)        #交叉熵
    #         loss.backward()
    #         optimizer.step()                          #更新權重
    #         running_loss += loss.item()
    
    #     print('Epoch [%d/%d], Loss: %.4f,need time %.4f'
    #             % (epoch + 1, num_epochs, running_loss / (4000 / batch_size), time.time() - batch_size_start))
    # torch.save(model, 'net.pth')  # 保存整個神經網絡的模型結構以及參數

                