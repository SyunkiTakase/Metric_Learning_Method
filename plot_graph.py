import csv
import argparse
import matplotlib.pyplot as plt

def plot_loss(epochs, train_losses, ce_losses, metric_losses, val_losses, save_path):
    loss_file = save_path + 'loss.png' 

    if train_losses or val_losses:
        # 学習曲線の描画
        plt.figure(figsize=(5, 5))

        # 損失のプロット
        if train_losses:
            plt.plot(epochs, train_losses, label='Train Loss')
        if ce_losses:
            plt.plot(epochs, ce_losses, label='CE Loss')
        if metric_losses:
            plt.plot(epochs, metric_losses, label='Metric Loss')
        if val_losses:
            plt.plot(epochs, val_losses, label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(loss_file)

    else:
        pass

def plot_acc(epochs, train_acc, val_acc, save_path):
    acc_file = save_path + 'acc.png' 

    if train_acc or val_acc:
        # 学習曲線の描画
        plt.figure(figsize=(5, 5))

        # 精度のプロット
        if train_acc:
            plt.plot(epochs, train_acc, label='Train Accuracy')
        if val_acc:
            plt.plot(epochs, val_acc, label='Val Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(acc_file)

    else:
        pass

def load_csv(args):
    csv_file = args.csv_path
    save_path = '/'.join(csv_file.split('/')[:-1]) + '/'

    epochs = []
    train_losses = []
    ce_losses = []
    metric_losses = []
    train_acc = []
    val_losses = []
    val_acc = []

    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            if row['Train Loss']:
                train_losses.append(float(row['Train Loss']))
            if row['CE Loss']:
                ce_losses.append(float(row['CE Loss']))
            if row['Metric Loss']:
                metric_losses.append(float(row['Metric Loss']))
            if row['Train Acc']:
                train_acc.append(float(row['Train Acc']))
            if row['Val Loss']:
                val_losses.append(float(row['Val Loss']))
            if row['Val Acc']:
                val_acc.append(float(row['Val Acc']))

    plot_loss(epochs, train_losses, ce_losses, metric_losses, val_losses, save_path)
    plot_acc(epochs, train_acc, val_acc, save_path)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./log.csv')
    args=parser.parse_args()

    load_csv(args)