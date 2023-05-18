import os
import torch
import torch.nn.functional as torchF
import matplotlib.pyplot as plt
import numpy as np
import glob
from Network import MyLSTM
from DataLoader import MyDataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

decay = 4e-5

def validating(model, validationset, kind="val"):
    with torch.no_grad():  # close grad tracking to reduce memory consumption

        total_correct = 0
        total_samples = 0
        model.eval()

        for batch in validationset:
            images, labels= batch
            
            preds = model(images)
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

            total_samples += len(labels)

        val_acc = total_correct / total_samples
        print(kind + "_acc: ", total_correct / total_samples)
        # get confusion matrix elements [(labels, preds),...]#model.train()return val_acc

    return val_acc

def main():
    file_path = 'C:/Users/何雨轩/Desktop/CSI程序/CSI程序/data1/ProcessedData2' #存放某一类动作测试集的文件夹
    Dataset = MyDataLoader(file_path)
    Dataset = torch.utils.data.DataLoader(Dataset, batch_size=32, shuffle=True)
    #
    # model = np.load('C:/Users/何雨轩/Desktop/CSI程序/CSI程序/data1/models/model.pth') #存放模型的路径

    model = torch.load('C:/Users/何雨轩/Desktop/CSI程序/CSI程序/data1/models/model.pth',map_location="cpu") #存放模型的路径
    model.eval()
    acc= validating(model, Dataset, kind="test")

if __name__ == '__main__':
    main()
