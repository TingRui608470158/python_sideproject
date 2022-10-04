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
from mask_data import MaskDataset
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable


if __name__ == '__main__':
    print("Training start")
    print("Torch version:",torch.__version__)
    ## cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/mask", help="image path")
    args = parser.parse_args()
    
    ##模型
    model = models.vgg16(pretrained=True)
    model.classifier.add_module("7", nn.Linear(1000, 3))
    model.classifier.add_module("8", nn.Softmax(dim = 1))
    model = model.to(device)
 
    print(model)
    print(model._modules.keys())

    ## 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    ## loss
    criterion = nn.CrossEntropyLoss().cuda()

    # 讀取資料集
    dataset = MaskDataset(args)
    
    # for i, path in enumerate(dataset.im_path):
    #     one_image = single_img(path)
    #     # one_image.show()
    #     print(dataset.im_path[i])
    #     print(max(dataset.im_label[i]))
    batch_size = 1
    dataloader = DataLoader(dataset=dataset, batch_size= batch_size, shuffle=True)
    # dataiter = iter(dataloader)
    # data = dataiter.next()
    #訓練
    model.train()
    num_epochs = 3
    for epoch in range(num_epochs):
        batch_size_start = time.time()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = Variable(inputs.float()).cuda()
            labels = Variable(labels).cuda()
            # labels = Variable(fun.one_hot(labels, num_classes=3)).cuda()
            inputs = inputs.permute(0, 3, 2, 1)
            # inputs = torch.unsqueeze(inputs,dim = 0)
            # print(i,inputs.shape,"\n label = ",labels)
            optimizer.zero_grad()
            
            outputs = model((inputs).to(device))

            loss = criterion(outputs, labels)        #交叉熵
            loss.backward()
            optimizer.step()                          #更新權重
            running_loss += loss.item()
    
        print('Epoch [%d/%d], Loss: %.4f,need time %.4f'
                % (epoch + 1, num_epochs, running_loss / (4000 / batch_size), time.time() - batch_size_start))
    torch.save(model, 'net.pth')  # 保存整個神經網絡的模型結構以及參數

                
        
    

#VGG16 架構
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )