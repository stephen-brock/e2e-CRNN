#!/usr/bin/env python3

#used Applied Deep Learning lab code as a template
 
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import dataset
import evaluation
from sklearn.metrics import roc_auc_score

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="CNN for training MagnaTune labeling",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# default_dataset_dir = "../MagnaTagATune/"
default_dataset_dir = "/mnt/storage/scratch/uq20042/MagnaTagATune/"

#arguments
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)

parser.add_argument("--learning-rate", default=7e-4, type=float, help="Learning rate")
parser.add_argument("--gamma", default=0.95, type=float, help="Gamma")

parser.add_argument("--spectrogram", action='store_true', default=False, help="Enable Spectrogram Transform")
parser.add_argument("--rnn", action='store_true', default=False,help="Use GRU RNN")
parser.add_argument("--dropout", action='store_true', default=False,help="Use Dropout")
parser.add_argument("--norm", action='store_true', default=False, help="Use Batch Normalisation")
parser.add_argument("--length", default=256, type=int, help="Length")
parser.add_argument("--stride", default=256, type=int, help="Stride")
parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=40,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=50,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=50,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=5,
    type=int,
    help="Number of worker processes used to load data.",
)
if parser.parse_args().spectrogram:
    print(parser.parse_args().spectrogram)
    from torchaudio.transforms import MelSpectrogram

#device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

#add noise transformation
#unused
def add_noise(x):
    return x + torch.randn(x.shape) / 2

def main(args):
    #initialise datasets
    if args.spectrogram:
        transform = MelSpectrogram(sample_rate=12000, n_fft=2048, win_length=args.length, hop_length=args.stride)
        train_dataset = dataset.MagnaTagATune(args.dataset_root + "annotations/train_labels.pkl", args.dataset_root + "samples/", transform=transform)
        train_validation = dataset.MagnaTagATune(args.dataset_root + "annotations/val_labels.pkl", args.dataset_root + "samples/", transform=transform)
    else:
        train_dataset = dataset.MagnaTagATune(args.dataset_root + "annotations/train_labels.pkl", args.dataset_root + "samples/")
        train_validation = dataset.MagnaTagATune(args.dataset_root + "annotations/val_labels.pkl", args.dataset_root + "samples/")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        train_validation,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True
    )

    #initialise model
    if args.spectrogram:
        model = CNN_S(args.length, args.stride, use_rnn = args.rnn, use_dropout = args.dropout, use_norm = args.norm)
    else:
        model = CNN(args.length, args.stride, use_rnn = args.rnn, use_dropout = args.dropout, use_norm = args.norm)
    
    #loss function
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, scheduler, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()



class CNN_S(nn.Module):
    def __init__(self, length, stride, use_rnn = True, use_dropout = True, use_norm = True, sub_clips: int = 10, channels: int = 1, sample_count: int = 34950):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        lOut = 138 * 256 / stride
        lOutF = 128
        padding = int(np.round(max(0, (61 - lOut) / 2)))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, padding=padding)
        self.initialise_layer(self.conv1)
        if use_norm:
            self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(4)
        
        if use_dropout:
            self.dropout1 = nn.Dropout(0.1)
        
        lOut = np.round((2 * padding + lOut - 9) / 4)
        lOutF = np.round((2 * padding + lOutF - 9) / 4)
        
        #second main CNN
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=8)
        if use_norm:
            self.norm2 = nn.BatchNorm2d(32)
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(4)
        
        if use_dropout:
            self.dropout2 = nn.Dropout(0.1)
        
        lOut = int(np.round((lOut - 9) / 4))
        lOutF = int(np.round((lOutF - 9) / 4))
        
        self.input_dim = lOut * lOutF * 32
        self.use_rnn = use_rnn
        #fully connected layers
        if use_rnn:
            self.gru1 = nn.GRU(self.input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
            if use_norm:
                self.norm3 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 50)
            if use_dropout:
                self.dropout3 = nn.Dropout(0.1)
        else:
            self.fc1 = nn.Linear(self.input_dim, 100)
            self.initialise_layer(self.fc1)
            if use_norm:
                self.norm3 = nn.BatchNorm1d(100)
            if use_dropout:
                self.dropout3 = nn.Dropout(0.4)
            
            self.fc2 = nn.Linear(100, 50)
        
        self.initialise_layer(self.fc2)
        
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        
        #flatten for batches + subclips in one dimension
        x = torch.flatten(samples, 0, 1)
        x = torch.log(1 + x * 10000)
        x = self.conv1(x)
        if self.use_norm:
            x = F.relu(self.norm1(x))
        else:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if self.use_norm:
            x = F.relu(self.norm2(x))
        else:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.pool2(x)
        
        if self.use_rnn:
            #reshape for RNN
            x = torch.flatten(x, start_dim=1)
            x = x.reshape(samples.shape[0], samples.shape[1], self.input_dim)
            
            x, _ = self.gru1(x)
            # take only last output of RNN
            x = x[...,-1:,:].squeeze()
            if self.use_norm:
                x = self.norm3(x)
            
            x = torch.sigmoid(self.fc2(x))
        else:
            #flatten for fully connected layer
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            if self.use_norm:
                x = F.relu(self.norm3(x))
            else:
                x = F.relu(x)
            if self.use_dropout:
                x = self.dropout3(x)
            
            x = torch.sigmoid(self.fc2(x))
            #reshape to get separate batches/subclip dimensions
            x = x.reshape(samples.shape[0], samples.shape[1], 50)
            
            #average over all subclips
            x = torch.mean(x, 1)
        
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class CNN(nn.Module):
    def __init__(self, length, stride, use_rnn = True, use_dropout = True, use_norm = True, sub_clips: int = 10, channels: int = 1, sample_count: int = 34950):
        super().__init__()
        self.use_rnn = use_rnn
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        #pad so theres at least one output remaining after convolutions
        padding = 56 * stride + length - sample_count
        padding = max(0, int(np.ceil(padding / 2)))
        
        self.conv0 = nn.Conv1d(in_channels=channels, out_channels=128, kernel_size=length, stride=stride,padding=padding)
        if use_norm:
            self.norm0 = nn.BatchNorm1d(128)
        if use_dropout:
            self.dropout0 = nn.Dropout(0.2)
        self.initialise_layer(self.conv0)
        
        lOut = np.ceil((sample_count + 2 * padding - length) / stride)
        
        #first main CNN
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=8)
        self.initialise_layer(self.conv1)
        if use_norm:
            self.norm1 = nn.BatchNorm1d(32)
        if use_dropout:
            self.dropout1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(4)
        
        
        lOut = np.ceil((lOut - 8) / 4)
        
        #second main CNN
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.initialise_layer(self.conv2)
        if use_norm:
            self.norm2 = nn.BatchNorm1d(32)
        if use_dropout:
            self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool1d(4)
        
        
        lOut = np.ceil((lOut - 8) / 4)
        
        self.input_dim = int(lOut) * 32
        if use_rnn:
            self.gru1 = nn.GRU(self.input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
            if use_norm:
                self.norm3 = nn.BatchNorm1d(128)
                
            self.fc2 = nn.Linear(128, 50)
        else:
            #fully connected layers
            #* 32 from flattening
            self.fc1 = nn.Linear(self.input_dim, 100)
            self.initialise_layer(self.fc1)
            
            if use_norm:
                self.norm3 = nn.BatchNorm1d(100)
            if use_dropout:
                self.dropout3 = nn.Dropout(0.4)
            
            self.fc2 = nn.Linear(100, 50)
        
        
        self.initialise_layer(self.fc2)
        
        

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        #flatten for batches + subclips in one dimension
        x = torch.flatten(samples, 0, 1)
        
        #normalize input between -1 and 1
        x = x / 32768
        
        x = self.conv0(x)
        if self.use_norm:
            x = F.relu(self.norm0(x))
        else:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout0(x)
        
        x = self.conv1(x)
        if self.use_norm:
            x = F.relu(self.norm1(x))
        else:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if self.use_norm:
            x = F.relu(self.norm2(x))
        else:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.pool2(x)
        
        if self.use_rnn:
            #reshape for RNN
            x = torch.flatten(x, start_dim=1)
            x = x.reshape(samples.shape[0], samples.shape[1], self.input_dim)
            x, _ = self.gru1(x)
            # take only last output of RNN
            x = x[...,-1:,:].squeeze()
            if self.use_norm:
                x = self.norm3(x)
            
            x = torch.sigmoid(self.fc2(x))
        else:
            #flatten for fully connected layer
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            if self.use_norm:
                x = F.relu(self.norm3(x))
            else:
                x = F.relu(x)
            if self.use_dropout:
                x = self.dropout3(x)
            
            x = torch.sigmoid(self.fc2(x))
            
            # #reshape to get separate batches/subclip dimensions
            x = x.reshape(samples.shape[0], samples.shape[1], 50)
            
            #average over all subclips
            x = torch.mean(x, 1)
        
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: optim.lr_scheduler,
        summary_writer: SummaryWriter,
        device: torch.device
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        results = {"preds": [], "labels": []}
        
        #training loop
        for epoch in range(start_epoch, epochs):
            self.model.train()
            for filenames, samples, labels in self.train_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                logits = self.model.forward(samples)
                
                loss = self.criterion(logits, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                
                    results["preds"].extend(list(logits.cpu().numpy()))
                    results["labels"].extend(list(labels.cpu().numpy()))
                    if ((self.step + 1) % log_frequency) == 0:
                        self.summary_writer.add_scalars(
                                "loss",
                                {"train": float(loss.item())},
                                self.step
                        )
                        try:
                            auc_score = roc_auc_score(y_true=results["labels"], y_score=results["preds"])
                            #other metrics
                            self.summary_writer.add_scalars(
                                    "AUC_SCORE",
                                    {"train": auc_score},
                                    self.step
                            )
                            results = {"preds": [], "labels": []}
                        except:
                            #missing classes, will accumulate data for next log
                            print("not enough data for roc_auc_score")
                    if ((self.step + 1) % print_frequency) == 0:
                        print("[", epoch, "]: ", loss)

                self.step += 1
            
            self.scheduler.step()
            #for translating step back to epoch
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
                
        #validate final trained model
        self.validate(True)

    def validate(self, print_stats = False):
        #set to evaluation
        self.model.eval()
        
        results = {"preds": [], "labels": []}
        total_loss = 0
        
        if print_stats:
            label_losses = torch.zeros((50))
            label_losses = label_losses.to(self.device)
            min_loss = 1000000000
            min_label = torch.zeros((50))
            min_pred = torch.zeros((50))
            min_filename = ""
            max_loss = -1
            max_label = torch.zeros((50))
            max_pred = torch.zeros((50))
            max_filename = ""
            
        
        steps = 0

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for filenames, samples, labels in self.val_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(samples)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                #MSE for individual label losses
                if print_stats:
                    error = (logits - labels).pow(2)
                    loss_amount = (labels * error).sum(0)
                    label_losses += loss_amount
                    for i in range(logits.shape[0]):
                        single_loss = error[i][:].sum(0)
                        if single_loss < min_loss:
                            min_loss = single_loss
                            min_pred = logits[i][:]
                            min_label = labels[i][:]
                            min_filename = filenames[i]
                        if single_loss > max_loss:
                            #worst
                            max_loss = single_loss
                            max_pred = logits[i][:]
                            max_label = labels[i][:]
                            max_filename = filenames[i]
                
                # labelSum = labels.sum(0)
                # label_losses += labelSum * loss.item()
                
                results["preds"].extend(list(logits.cpu().numpy()))
                results["labels"].extend(list(labels.cpu().numpy()))
                steps = steps + 1

        average_loss = total_loss / len(self.val_loader)
        auc_score = roc_auc_score(y_true=results["labels"], y_score=results["preds"])
        print('TEST AUC Score: {}'.format(auc_score))
        print()
        print("-------------------------------------------------------------")

        #other metrics
        self.summary_writer.add_scalars(
                "AUC_SCORE",
                {"test": auc_score},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}")
        if print_stats:
            label_losses = label_losses / len(self.val_loader)
            #print per label losses
            print("Label losses: ", label_losses)
            print("Min Label: ", min_label)
            print("Min Pred: ", min_pred)
            print("Min Loss: ", min_loss)
            print("Min File: ", min_filename)
            print("Max Label: ", max_label)
            print("Max Pred: ", max_pred)
            print("Max Loss: ", max_loss)
            print("Max File: ", max_filename)
            self.summary_writer.add_histogram(
                    "Label Losses",
                    label_losses,
                    self.step
            )

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_len={args.length}_stride={args.stride}_spectro={args.spectrogram}_norm={args.norm}_do={args.dropout}_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())