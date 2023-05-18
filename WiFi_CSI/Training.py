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


def train_LSTM(model, Train_dataset, Validate_dataset=None, epoches=50, learningRate=0.001, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), learningRate * 3, weight_decay=decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, verbose=True, min_lr=0.000001)
    model.train()
    croloss = torch.nn.CrossEntropyLoss()

    for epoch in range(epoches):
        loss_total = 0
        for batch in Train_dataset:
            data, label = batch

            # image = image.cuda()
            # label = label.cuda()

            y_pre = model(data)

            loss = croloss(y_pre.squeeze(1), label)
            loss_total += loss.item()
            print("loss: ", loss.item())
            ####
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step(loss_total)
        print('Epoch:', epoch + 1)
        print("total loss: ", loss_total)

        torch.save(model, 'C:/Users/何雨轩/Desktop/CSI程序/CSI程序/修改程序/WiFi_CSI/models/model/model.pth')
        print('----------------Model saved------------------')

    return model


def main():
    file_path = 'C:/Users/何雨轩/Desktop/CSI程序/CSI程序/data1/ProcessedData/'

    Dataset = MyDataLoader(file_path)

    ### split dataset
    batch_size = 8
    train_size = int(len(Dataset) * 0.8)  # 80%作训练集
    validation_size = len(Dataset) - train_size  # 20%作验证集

    train_set_split, validation_set_split = torch.utils.data.random_split(Dataset, [train_size, validation_size])
    ### load dataset

    train_dataset = torch.utils.data.DataLoader(train_set_split, batch_size=batch_size, shuffle=True)
    validation_dataset = torch.utils.data.DataLoader(validation_set_split, batch_size=batch_size, shuffle=True)

    ###start training

    model = MyLSTM()
    model = train_LSTM(model, train_dataset, validation_dataset,
                       epoches=40, learningRate=0.0005, batch_size=batch_size)


if __name__ == '__main__':
    main()
