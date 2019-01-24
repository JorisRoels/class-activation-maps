
import datetime
import os
import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import cv2

from networks.blocks import UNetConvBlock2D, UNetUpSamplingBlock2D, UNetConvBlock3D, UNetUpSamplingBlock3D
from util.metrics import jaccard, accuracy_metrics

UNSUPERVISED_TRAINING = 0
SUPERVISED_TRAINING = 1

# original 2D unet encoder
class UNetEncoder2D(nn.Module):

    def __init__(self, in_channels=1, feature_maps=64, levels=4, group_norm=True):
        super(UNetEncoder2D, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock2D(in_features, out_features, group_norm=group_norm)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock2D(2**(levels-1) * feature_maps, 2**levels * feature_maps)

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features,'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features,'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs

# original 3D unet encoder
class UNetEncoder3D(nn.Module):

    def __init__(self, in_channels=1, feature_maps=64, levels=4, group_norm=True):
        super(UNetEncoder3D, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock3D(in_features, out_features, group_norm=group_norm)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock3D(2**(levels-1) * feature_maps, 2**levels * feature_maps)

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features,'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features,'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs

# original 2D unet decoder
class UNetDecoder2D(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, skip_connections=True, group_norm=True, pretrain_unsupervised=False, cam_maps=256):
        super(UNetDecoder2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cam_maps = cam_maps
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.pretrain_unsupervised = pretrain_unsupervised
        self.phase = SUPERVISED_TRAINING
        if pretrain_unsupervised:
            self.phase = UNSUPERVISED_TRAINING
        self.features = nn.Sequential()

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock2D(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections:
                conv_block = UNetConvBlock2D(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, group_norm=group_norm)
            else:
                conv_block = UNetConvBlock2D(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps, group_norm=group_norm)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # activation maps
        self.activations = nn.Conv2d(feature_maps, cam_maps, 3, padding=1)

        # classification layer
        self.out = nn.Linear(cam_maps, out_channels)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                if self.phase == SUPERVISED_TRAINING:
                    outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[i], outputs)  # also deals with concat
                else:
                    outputs = getattr(self.features, 'upconv%d' % (i + 1))(torch.zeros_like(encoder_outputs[i]), outputs)  # also deals with concat
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        # CAM activations
        h = self.activations(outputs)

        # global average pooling
        h = F.avg_pool2d(h, (h.size()[2], h.size()[3])).view(-1, self.cam_maps)

        # class output
        outputs = self.out(h)

        if self.phase == SUPERVISED_TRAINING:
            return decoder_outputs, outputs
        else:
            return decoder_outputs, outputs[:, 0:self.in_channels, :, :]

# original 3D unet decoder
class UNetDecoder3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, skip_connections=True, group_norm=True, pretrain_unsupervised=False):
        super(UNetDecoder3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.pretrain_unsupervised = pretrain_unsupervised
        self.phase = SUPERVISED_TRAINING
        if pretrain_unsupervised:
            self.phase = UNSUPERVISED_TRAINING
        self.features = nn.Sequential()

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock3D(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections:
                conv_block = UNetConvBlock3D(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, group_norm=group_norm)
            else:
                conv_block = UNetConvBlock3D(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps, group_norm=group_norm)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv3d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                if self.phase == SUPERVISED_TRAINING:
                    outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[i], outputs)  # also deals with concat
                else:
                    outputs = getattr(self.features, 'upconv%d' % (i + 1))(torch.zeros_like(encoder_outputs[i]), outputs)  # also deals with concat
            else:
                outputs = getattr(self.features,'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        if self.phase == SUPERVISED_TRAINING:
            return decoder_outputs, outputs
        else:
            return decoder_outputs, outputs[:, 0:self.in_channels, :, :]

# original 2D unet model
class UNet2D(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=True, pretrain_unsupervised=False, cam_maps=256):
        super(UNet2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cam_maps = cam_maps
        self.feature_maps = feature_maps
        self.levels = levels

        # contractive path
        self.encoder = UNetEncoder2D(in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)
        # expansive path
        self.decoder = UNetDecoder2D(in_channels, out_channels, feature_maps=feature_maps, levels=levels, skip_connections=True, group_norm=group_norm, pretrain_unsupervised=pretrain_unsupervised, cam_maps=cam_maps)

    def forward(self, inputs):

        # contractive path
        encoder_outputs, final_output = self.encoder(inputs)

        # expansive path
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)

        return outputs

    def get_cam(self, x, idx=1):

        # contractive path
        encoder_outputs, final_output = self.encoder(x)

        # expansive path
        inputs = final_output
        encoder_outputs.reverse()
        outputs = inputs
        for i in range(self.levels):
            if self.decoder.skip_connections:
                if self.decoder.phase == SUPERVISED_TRAINING:
                    outputs = getattr(self.decoder.features, 'upconv%d' % (i + 1))(encoder_outputs[i], outputs)  # also deals with concat
                else:
                    outputs = getattr(self.decoder.features, 'upconv%d' % (i + 1))(torch.zeros_like(encoder_outputs[i]), outputs)  # also deals with concat
            else:
                outputs = getattr(self.decoder.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.decoder.features, 'convblock%d' % (i + 1))(outputs)

        # produce cam units
        cam_units = self.decoder.activations(outputs)

        bs, nc, h, w = cam_units.shape

        cams = torch.zeros(bs, 1, h, w)
        for b in range(bs):
            cam = self.decoder.out.weight[idx:idx+1].data.cpu().numpy().dot(cam_units[b,...].reshape((nc, h * w)).data.cpu().numpy())
            cams[b, 0, ...] = torch.Tensor(cam.reshape(h, w))
        cams = nn.functional.interpolate(cams, (x.size(2), x.size(3)), mode="bilinear")
        cams = cams - cams.min()
        cams = cams / cams.max()

        return cams

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            if self.decoder.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.decoder.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
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
            if self.phase == SUPERVISED_TRAINING:
                writer.add_scalar('train/loss-seg', loss_avg, epoch)
            else:
                writer.add_scalar('train/loss-rec', loss_avg, epoch)
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                writer.add_image('train/x-rec-input', x, epoch)
                writer.add_image('train/x-rec-output', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

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
            if self.decoder.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.decoder.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            if self.decoder.phase == SUPERVISED_TRAINING:
                # compute other interesting metrics
                y_ = F.softmax(y_pred, dim=1).data.cpu().numpy()[:,1,...]
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
            if self.phase == SUPERVISED_TRAINING:
                cams = self.get_cam(x)
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                cams = vutils.make_grid(cams, normalize=True)
                writer.add_image('test/x', x, epoch)
                writer.add_image('test/x-cam', cams, epoch)
                writer.add_scalar('test/loss-seg', loss_avg, epoch)
                writer.add_scalar('test/jaccard', j_avg, epoch)
                writer.add_scalar('test/accuracy', a_avg, epoch)
                writer.add_scalar('test/precision', p_avg, epoch)
                writer.add_scalar('test/recall', r_avg, epoch)
                writer.add_scalar('test/f-score', f_avg, epoch)
            else:
                writer.add_scalar('test/loss-rec', loss_avg, epoch)
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                writer.add_image('test/x-rec-input', x, epoch)
                writer.add_image('test/x-rec-output', x_rec, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn_seg, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1,
                  train_loader_unsupervised=None, test_loader_unsupervised=None, loss_fn_rec=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        if self.decoder.pretrain_unsupervised:

            print('[%s] Starting unsupervised pre-training' % (datetime.datetime.now()))

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader_unsupervised, loss_fn=loss_fn_rec, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader_unsupervised, loss_fn=loss_fn_rec, epoch=epoch, writer=writer)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint_rec.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint_rec.pytorch'))

        print('[%s] Starting supervised pre-training' % (datetime.datetime.now()))
        self.phase = SUPERVISED_TRAINING

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_fn=loss_fn_seg, optimizer=optimizer, epoch=epoch,
                             print_stats=print_stats, writer=writer)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn_seg, epoch=epoch, writer=writer)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

# original 3D unet model
class UNet3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=True, pretrain_unsupervised=False):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels

        # contractive path
        self.encoder = UNetEncoder3D(in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)
        # expansive path
        self.decoder = UNetDecoder3D(in_channels, out_channels, feature_maps=feature_maps, levels=levels, skip_connections=True, group_norm=group_norm, pretrain_unsupervised=pretrain_unsupervised)

    def forward(self, inputs):

        # contractive path
        encoder_outputs, final_output = self.encoder(inputs)

        # expansive path
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)

        return outputs

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            if self.decoder.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.decoder.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
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
            if self.decoder.phase == SUPERVISED_TRAINING:
                writer.add_scalar('train/loss-seg', loss_avg, epoch)
            else:
                writer.add_scalar('train/loss-rec', loss_avg, epoch)

            if write_images:
                # write images
                x = x[:,:,x.size(2)//2,...]
                y_pred = y_pred[:,:,y_pred.size(2)//2,...]
                if self.decoder.phase == SUPERVISED_TRAINING:
                    y = y[:,:,y.size(2)//2,...]
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                    y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                    writer.add_image('train/x', x, epoch)
                    writer.add_image('train/y', y, epoch)
                    writer.add_image('train/y_pred', y_pred, epoch)
                else:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                    writer.add_image('train/x-rec-input', x, epoch)
                    writer.add_image('train/x-rec-output', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

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
            if self.decoder.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.decoder.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            if self.decoder.phase == SUPERVISED_TRAINING:
                # compute other interesting metrics
                y_ = F.softmax(y_pred, dim=1).data.cpu().numpy()[:,1,...]
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
            if self.decoder.phase == SUPERVISED_TRAINING:
                writer.add_scalar('train/loss-seg', loss_avg, epoch)
                writer.add_scalar('test/loss', loss_avg, epoch)
                writer.add_scalar('test/jaccard', j_avg, epoch)
                writer.add_scalar('test/accuracy', a_avg, epoch)
                writer.add_scalar('test/precision', p_avg, epoch)
                writer.add_scalar('test/recall', r_avg, epoch)
                writer.add_scalar('test/f-score', f_avg, epoch)
                if write_images:
                    # write images
                    x = x[:,:,x.size(2)//2,...]
                    y_pred = y_pred[:,:,y_pred.size(2)//2,...]
                    if self.decoder.phase == SUPERVISED_TRAINING:
                        y = y[:,:,y.size(2)//2,...]
                        x = vutils.make_grid(x, normalize=True, scale_each=True)
                        y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                        y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                        writer.add_image('test/x', x, epoch)
                        writer.add_image('test/y', y, epoch)
                        writer.add_image('test/y_pred', y_pred, epoch)
            else:
                writer.add_scalar('test/loss-rec', loss_avg, epoch)
                if write_images:
                    # write images
                    x = vutils.make_grid(x[:,:,x.size(2)//2,...], normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred[:,:,y_pred.size(2)//2,...]).data)
                    writer.add_image('test/x-rec-input', x, epoch)
                    writer.add_image('test/x-rec-output', x_rec, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn_seg, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1,
                  train_loader_unsupervised=None, test_loader_unsupervised=None, loss_fn_rec=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        if self.decoder.pretrain_unsupervised:

            print('[%s] Starting unsupervised pre-training' % (datetime.datetime.now()))

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader_unsupervised, loss_fn=loss_fn_rec, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader_unsupervised, loss_fn=loss_fn_rec, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint_rec.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch_rec'))

            print('[%s] Starting supervised pre-training' % (datetime.datetime.now()))
            self.decoder.phase = SUPERVISED_TRAINING

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader, loss_fn=loss_fn_seg, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn_seg, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

            writer.close()

# 2D unet autoencoder model
class Autoencoder2D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, feature_maps=64, levels=4, group_norm=True, sigma_min=0.0, sigma_max=1.0):
        super(Autoencoder2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # contractive path
        self.encoder = UNetEncoder2D(in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)
        # expansive path
        self.decoder = UNetDecoder2D(out_channels, feature_maps=feature_maps, levels=levels, skip_connections=False, group_norm=group_norm)

    def forward(self, inputs):

        # contractive path
        encoder_outputs, final_output = self.encoder(inputs)

        # expansive path
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)

        return outputs

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            x = data[0].cuda()

            # add noise to the input (denoising autoencoder)
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)
            x_noise = x + sigma*torch.rand_like(x)

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_rec = self(x_noise)

            # compute loss
            loss = loss_fn(x_rec, x)
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
            writer.add_scalar('train/loss', loss_avg, epoch)

            if write_images:
                # write images
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                x_rec = vutils.make_grid(torch.sigmoid(x_rec).data)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/x_rec', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            x = data[0].cuda()

            # forward prop
            x_rec = self(x)

            # compute loss
            loss = loss_fn(x_rec, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss', loss_avg, epoch)

            if write_images:
                # write images
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                x_rec = vutils.make_grid(torch.sigmoid(x_rec).data)
                writer.add_image('test/x', x, epoch)
                writer.add_image('test/x_rec', x_rec, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

# 3D unet autoencoder model
class Autoencoder3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, feature_maps=64, levels=4, group_norm=True, sigma_min=0.0, sigma_max=1.0):
        super(Autoencoder3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # contractive path
        self.encoder = UNetEncoder3D(in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)
        # expansive path
        self.decoder = UNetDecoder3D(out_channels, feature_maps=feature_maps, levels=levels, skip_connections=False, group_norm=group_norm)

    def forward(self, inputs):

        # contractive path
        encoder_outputs, final_output = self.encoder(inputs)

        # expansive path
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)

        return outputs

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            x = data[0].cuda()

            # add noise to the input (denoising autoencoder)
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)
            x_noise = x + sigma*torch.rand_like(x)

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_rec = self(x)

            # compute loss
            loss = loss_fn(x_rec, x)
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
            writer.add_scalar('train/loss', loss_avg, epoch)

            if write_images:
                # write images
                x = x[:,:,x.size(2)//2,...]
                x_rec = x_rec[:,:,x_rec.size(2)//2,...]
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                x_rec = vutils.make_grid(torch.sigmoid(x_rec).data, scale_each=True)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/x_rec', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            x = data[0].cuda()

            # forward prop
            x_rec = self(x)

            # compute loss
            loss = loss_fn(x_rec, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss', loss_avg, epoch)

            if write_images:
                # write images
                x = x[:,:,x.size(2)//2,...]
                x_rec = x_rec[:,:,x_rec.size(2)//2,...]
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                x_rec = vutils.make_grid(torch.sigmoid(x_rec).data, scale_each=True)
                writer.add_image('test/x', x, epoch)
                writer.add_image('test/x_rec', x_rec, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()