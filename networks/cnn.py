
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import cv2

from util.metrics import jaccard, accuracy_metrics

UNSUPERVISED_TRAINING = 0
SUPERVISED_TRAINING = 1

# original 2D CNN model
class CNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, pretrain_unsupervised=False):
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrain_unsupervised = pretrain_unsupervised
        self.phase = SUPERVISED_TRAINING

        # conv1
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 48, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(48, 48, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(48, 48, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(48, 48, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(48, 48, 5, padding=2)
        )

        if pretrain_unsupervised:

            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(48, 48, 5, padding=2), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(48, 48, 5, padding=2), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(48, 48, 5, padding=2), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(48, self.in_channels, 5, padding=2)
            )

        # fc5
        self.fc5 = nn.Linear(6*6*48, 200)
        self.relu5 = nn.ReLU(inplace=True)

        # fc6
        self.fc6 = nn.Linear(200, out_channels)

    def forward(self, x):

        h = x

        h = self.encoder(h)

        if self.pretrain_unsupervised:

            h = self.decoder(h)

        else:

            h = h.view(h.size(0),-1)

            h = self.relu5(self.fc5(h))

            h = self.fc6(h)

        return h

    # trains the network for one epoch
    def train_unsupervised_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            x = data.cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics if necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader.dataset), loss))

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average train loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('train/loss-rec', loss_avg, epoch)
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
            writer.add_image('train/x-rec-input', x, epoch)
            writer.add_image('train/x-rec-output', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_unsupervised_epoch(self, loader, loss_fn, epoch, writer=None):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            x = data.cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss-rec', loss_avg, epoch)
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
            writer.add_image('test/x-rec-input', x, epoch)
            writer.add_image('test/x-rec-output', x_rec, epoch)

        return loss_avg

    # trains the network for one epoch
    def train_supervised_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics if necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader.dataset), loss))

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average train loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('train/loss-seg', loss_avg, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_supervised_epoch(self, loader, loss_fn, epoch, writer=None):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        j_cum = 0.0
        a_cum = 0.0
        p_cum = 0.0
        r_cum = 0.0
        f_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # compute other interesting metrics
            y_ = F.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
            j_cum += jaccard(y_, y.cpu().numpy())
            a, p, r, f = accuracy_metrics(y_, y.cpu().numpy())
            a_cum += a; p_cum += p; r_cum += r; f_cum += f

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        j_avg = j_cum / cnt
        a_avg = a_cum / cnt
        p_avg = p_cum / cnt
        r_avg = r_cum / cnt
        f_avg = f_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss-seg', loss_avg, epoch)
            writer.add_scalar('test/jaccard', j_avg, epoch)
            writer.add_scalar('test/accuracy', a_avg, epoch)
            writer.add_scalar('test/precision', p_avg, epoch)
            writer.add_scalar('test/recall', r_avg, epoch)
            writer.add_scalar('test/f-score', f_avg, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        if self.pretrain_unsupervised:

            print('[%s] Starting unsupervised pre-training' % (datetime.datetime.now()))

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_unsupervised_epoch(loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_unsupervised_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint_rec.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint_rec.pytorch'))

        else:

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_supervised_epoch(loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_supervised_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()