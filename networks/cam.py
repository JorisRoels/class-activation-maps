
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from util.metrics import jaccard, accuracy_metrics

class ModelCAM(nn.Module):

    def __init__(self, feature_model, feature_channels, n_units=1024, num_classes=2):
        super(ModelCAM, self).__init__()

        self.feature_model = feature_model
        self.feature_channels = feature_channels
        self.n_units = n_units

        # cam units
        self.units = nn.Conv2d(feature_channels, n_units, kernel_size=3, padding=1)

        # classification layer
        self.out = nn.Linear(n_units, num_classes)

    def forward(self, x):

        # compute base network features
        features = self.feature_model(x)

        # produce cam units
        cam_units = self.units(features)

        # perform global average pooling and flatten
        gap_units = F.avg_pool2d(cam_units, (features.size()[2],features.size()[3])).view(-1,self.n_units)

        # final layer does the classification
        return self.out(gap_units)

    def get_cam(self, x, idx=1, rescale=True, normalize=True):

        # compute base network features
        features = self.feature_model(x)

        # produce cam units
        cam_units = self.units(features)

        # compute cams
        bs, nc, h, w = cam_units.shape
        cams = torch.zeros(bs, 1, h, w)
        for b in range(bs):
            cam = self.out.weight[idx:idx+1].data.cpu().numpy().dot(cam_units[b,...].reshape((nc, h * w)).data.cpu().numpy())
            cams[b, 0, ...] = torch.Tensor(cam.reshape(h, w))

        if rescale:
            cams = F.interpolate(cams, (x.size(2), x.size(3)), mode="bilinear")

        if normalize:
            cams = cams - cams.min()
            cams = cams / cams.max()

        return cams

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

            # log cams
            cams = self.get_cam(x)
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            cams = vutils.make_grid(cams, normalize=True)
            writer.add_image('train/x', x, epoch)
            writer.add_image('train/x-cam', cams, epoch)

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

            # log cams
            cams = self.get_cam(x)
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            cams = vutils.make_grid(cams, normalize=True)
            writer.add_image('test/x', x, epoch)
            writer.add_image('test/x-cam', cams, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

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
