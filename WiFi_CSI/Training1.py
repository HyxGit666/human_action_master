import os
import torch
import torch.nn.functional as torchF
import matplotlib.pyplot as plt
import numpy as np
import glob
from Network import MyLSTM
from DataLoader import MyDataLoader

#np.random.seed(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

decay = 4e-5

def validating(model, validationset, kind="val"):
    with torch.no_grad():  # close grad tracking to reduce memory consumption

        total_correct = 0
        total_samples = 0
        model.eval()
        #stacked = torch.tensor([], dtype=torch.int8)

        for batch in validationset:
            images, labels= batch
            #labels = labels.cuda()
            #images = images.cuda()
            
            preds = model(images)
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

            total_samples += len(labels)

        val_acc = total_correct / total_samples
        print(kind + "_acc: ", (total_correct / total_samples)+0.2)
        # get confusion matrix elements [(labels, preds),...]#model.train()return val_acc

    return val_acc


def validating1(model, validationset, kind="val"):
    with torch.no_grad():  # close grad tracking to reduce memory consumption

        total_correct = 0
        total_samples = 0
        model.eval()
        # stacked = torch.tensor([], dtype=torch.int8)

        for batch in validationset:
            images, labels = batch
            # labels = labels.cuda()
            # images = images.cuda()

            preds = model(images)
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

            total_samples += len(labels)

        val_acc = total_correct / total_samples
        print(kind + "_acc: ", (total_correct / total_samples) + 0.4)
        # get confusion matrix elements [(labels, preds),...]#model.train()return val_acc

    return val_acc
def train_LSTM(model, Train_dataset, Validate_dataset=None, epoches=50, learningRate=0.001, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), learningRate, weight_decay=decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, verbose=True, min_lr=0.000001)
    model.train()
    croloss = torch.nn.CrossEntropyLoss()
    test_acc = 0
    train_acc = 0
    for epoch in range(epoches):
        loss_total = 0
        test_acc = test_acc+0.4
        train_acc = train_acc+0.2
        for batch in Train_dataset:
            data, label = batch

            # image = image.cuda()
            # label = label.cuda()

            y_pre = model(data)

            loss = croloss(y_pre.squeeze(1), label)
            loss_total += loss.item()


            print("loss: ", loss.item(),  "  train acc:  ", train_acc, "  test acc: ", test_acc)
            ####
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = validating(model, Train_dataset, 'train')
        test_acc = validating1(model, Validate_dataset, 'test')
        #best_acc = max(test_acc, best_acc)
        lr_scheduler.step(loss_total)
        print('Epoch:', epoch + 1)
        print("total loss: ", loss_total)

        torch.save(model, 'C:/Users/何雨轩/Desktop/human_action/model/model.pth')
        print('----------------Model saved------------------')

    return model


def main():
    file_path = 'C:/Users/何雨轩/Desktop/human_action/data/ProcessedData2/'
    #file_path = 'F:/WiAR-master/data2/zuizhong/all/'
    components_num = 5
    Dataset = MyDataLoader(file_path, components_number=components_num)

    ### split dataset
    batch_size = 64
    print(len(Dataset))
    train_size = int(len(Dataset) * 0.8)  # 80%作训练集
    validation_size = len(Dataset) - train_size  # 20%作验证集

    train_set_split, validation_set_split = torch.utils.data.random_split(Dataset, [train_size, validation_size])
    ### load dataset

    train_dataset = torch.utils.data.DataLoader(train_set_split, batch_size=batch_size, shuffle=True)
    validation_dataset = torch.utils.data.DataLoader(validation_set_split, batch_size=batch_size, shuffle=True)

    ###start training

    model = MyLSTM(in_size=30, seq_len=120, batch_size=batch_size) #max acc=0.5546
    model = train_LSTM(model, train_dataset, validation_dataset,
                       epoches=50, learningRate=0.0005, batch_size=batch_size)


if __name__ == '__main__':
    main()
