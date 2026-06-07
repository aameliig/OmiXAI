# imports
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_func(output, y_batch):
    return torch.nn.NLLLoss()(torch.transpose(output, 2, 1), y_batch)


def train_epoch(model, optimizer, loader_train):
    roc_auc_log, precision_log, recall_log, f1_log, acc_log, loss_log = [], [], [], [], [], []
    model.train()
    for X_batch, y_batch in loader_train:
        device = torch.device('cuda')
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
        optimizer.zero_grad()
        output = model(X_batch)
        if output.dim() == 2:
            output = output.unsqueeze(0)
        pred = torch.argmax(output, dim=2)
        with torch.no_grad():
            y_pred = nn.Softmax(dim=1)(output)[:, :,1].cpu().numpy().flatten()
            if np.std(y_batch.cpu().numpy().flatten()) == 0:
                roc_auc = 0
                precision = 0
                recall = 0
            else:
                roc_auc = roc_auc_score(y_batch.cpu().numpy().flatten(),
                                        y_pred)

                precision = precision_score(y_batch.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), zero_division=0)
                recall = recall_score(y_batch.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

            precision_log.append(precision)
            recall_log.append(recall)
            f1_log.append(f1_score(y_batch.cpu().numpy().flatten(),
                             pred.cpu().numpy().flatten(), zero_division=0))

        roc_auc_log.append(roc_auc)
        acc = torch.mean((pred.to(device) == y_batch).float())
        acc_log.append(acc.cpu().numpy())
        loss = loss_func(output, y_batch)
        loss.backward()
        optimizer.step()

        loss = loss.item()
        loss_log.append(loss)
        torch.cuda.empty_cache()
    return roc_auc_log, precision_log, recall_log, f1_log, acc_log, loss_log

def test(model, loader_test):
    np.random.seed(42)
    roc_auc_log, precision_log, recall_log, f1_log, acc_log, loss_log = [], [], [], [], [], []
    model.eval()
    means = []
    with torch.no_grad():
        for X_batch, y_batch in loader_test:
            device = torch.device('cuda')
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
            #X_batch, y_batch = X_batch.cuda(), y_batch.cuda().long()
            output = model(X_batch)

            if output.dim() == 2:
                output = output.unsqueeze(0)
            means.append(y_batch.sum().cpu() / (1.0 * y_batch.shape[0]))
            pred = torch.argmax(output, dim=2)
            if np.std(y_batch.cpu().numpy().flatten()) == 0:
                roc_auc = 0
                precision = 0
                recall = 0
            else:
                roc_auc = roc_auc_score(y_batch.cpu().numpy().flatten(),
                                        nn.Softmax(dim=1)(output)[:, :,1].detach().cpu().numpy().flatten())
                precision = precision_score(y_batch.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), zero_division=0)
                recall = recall_score(y_batch.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

            f1 = f1_score(y_batch.cpu().numpy().flatten(),
                                      pred.cpu().numpy().flatten(), zero_division=0)
            if f1 == 0.0 and torch.all(y_batch == 0) and torch.all(pred == 0):
                pass
            else:
                f1_log.append(f1)
            roc_auc_log.append(roc_auc)
            precision_log.append(precision)
            recall_log.append(recall)
            acc = torch.mean((pred.to(device) == y_batch).float())
            acc_log.append(acc.cpu().numpy())
            loss = loss_func(output, y_batch)
            loss_log.append(loss.item())
            torch.cuda.empty_cache()
    return roc_auc_log, precision_log, recall_log, f1_log, acc_log, loss_log

def plot_history(train_history, valid_history, title, BatchSize, epoch_to_show=20):
    plt.figure(figsize=(epoch_to_show, 4))
    plt.title(title)

    epoch_num = len(valid_history)
    train_history = np.array([None] * (BatchSize * epoch_to_show) + train_history)
    valid_history = np.array([None] * epoch_to_show + valid_history)

    plt.plot(np.linspace(epoch_num-epoch_to_show+1, epoch_num+1, (epoch_to_show+1)*BatchSize),
             train_history[-(epoch_to_show+1)*BatchSize:], c='red', label='train')
    plt.plot(np.linspace(epoch_num-epoch_to_show+1, epoch_num+1, epoch_to_show+1),
                valid_history[-epoch_to_show-1:], c='green', label='test')

    plt.ylim((0, 1))
    plt.yticks(np.linspace(0, 1, 11))
    plt.xticks(np.arange(epoch_num-epoch_to_show+1, epoch_num+2),
              np.arange(epoch_num-epoch_to_show, epoch_num+1).astype(int))
    plt.xlabel('train steps')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def train(model, opt, n_epochs, loader_train, loader_test):
    train_auc_log, train_pr_log, train_rec_log, train_f1_log, train_acc_log, train_loss_log = [], [], [], [], [], []
    val_auc_log, val_pr_log, val_rec_log, val_f1_log, val_acc_log, val_loss_log = [], [], [], [], [], []
    times = []
    best_models = []
    for epoch in range(n_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch + 1, n_epochs))
        train_auc, train_pr, train_rec, train_f1, train_acc, train_loss = train_epoch(model, opt, loader_train)
        val_auc, val_pr, val_rec, val_f1, val_acc, val_loss = test(model, loader_test)

        best_models.append(deepcopy(model))

        end_time = time.time()
        times.append(end_time - start_time)
        BatchSize = len(train_loss)

        train_auc_log.extend(train_auc)
        train_pr_log.extend(train_pr)
        train_rec_log.extend(train_rec)
        train_f1_log.extend(train_f1)
        train_acc_log.extend(train_acc)
        train_loss_log.extend(train_loss)

        val_auc_log.append(np.mean(val_auc))
        val_pr_log.append(np.mean(val_pr))
        val_rec_log.append(np.mean(val_rec))
        val_f1_log.append(np.mean(val_f1))
        val_acc_log.append(np.mean(val_acc))
        val_loss_log.append(np.mean(val_loss))

        if (epoch % 1) == 0:
            clear_output()
            plot_history(train_loss_log,     val_loss_log,     'Loss',     BatchSize)
            plot_history(train_acc_log, val_acc_log, 'Accuracy', BatchSize)
            plot_history(train_auc_log, val_auc_log, 'Auc',      BatchSize)
            plot_history(train_f1_log, val_f1_log,   'F1',       BatchSize)
            print("Time: ", end_time/60)
            print("Epoch {}: ROC-AUC = {:.2%}".format(epoch+1, val_auc_log[-1]))
            print("Epoch {}: Precision = {:.3}".format(epoch+1, val_pr_log[-1]))
            print("Epoch {}: Recall = {:.3}".format(epoch+1, val_rec_log[-1]))
            print("Epoch {}: F1-score = {:.3}".format(epoch+1, val_f1_log[-1]))


    print("Final ROC-AUC = {:.4}%".format(val_auc_log[-1]*100))
    print("Final Precision = {:.3}".format(val_pr_log[-1]))
    print("Final Recall = {:.3}".format(val_rec_log[-1]))
    print("Final F1-score = {:.3}".format(val_f1_log[-1]))

    return val_auc_log, val_pr_log, val_rec_log, val_f1_log, val_acc_log, val_loss_log, times, best_models
