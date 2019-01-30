
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from util.metrics import jaccard, accuracy_metrics
from util.validation import gaussian_window, sliding_window

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

    def segment_cam(self, data, input_shape, idx=1, rescale=True, normalize=True, batch_size=1, in_channels=1, step_size=None):

        # make sure we compute everything on the gpu and in evaluation mode
        self.cuda()
        self.eval()

        # 2D or 3D
        is2d = len(input_shape) == 2

        # upsampling might be necessary depending on the network
        interp = nn.Upsample(size=input_shape, mode='bilinear', align_corners=True)

        # set step size to half of the window if necessary
        if step_size == None:
            if is2d:
                step_size = (1, input_shape[0]//2, input_shape[1]//2)
            else:
                step_size = (input_shape[0]//2, input_shape[1]//2, input_shape[2]//2)

        # gaussian window for smooth block merging
        if is2d:
            g_window = gaussian_window((1,input_shape[0],input_shape[1]), sigma=input_shape[-1]/4)
        else:
            g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)

        # symmetric extension only necessary along z-axis if multichannel 2D inputs
        if is2d and in_channels>1:
            z_pad = in_channels // 2
            padding = ((z_pad, z_pad), (0, 0), (0, 0))
            data = np.pad(data, padding, mode='symmetric')
        else:
            z_pad = 0

        # allocate space
        seg_cum = np.zeros(data.shape)
        counts_cum = np.zeros(data.shape)

        # define sliding window
        if is2d:
            sw = sliding_window(data, step_size=step_size, window_size=(in_channels, input_shape[0],input_shape[1]))
        else:
            sw = sliding_window(data, step_size=step_size, window_size=input_shape)

        # start prediction
        batch_counter = 0
        if is2d:
            batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1]))
        else:
            batch = np.zeros((batch_size, in_channels, input_shape[0], input_shape[1], input_shape[2]))
        positions = np.zeros((batch_size, 3), dtype=int)
        for (z, y, x, inputs) in sw:

            # fill batch
            if not is2d: # add channel in case of 3D processing, in 2D case, it's already there
                inputs = inputs[np.newaxis, ...]
            batch[batch_counter, ...] = inputs
            positions[batch_counter, :] = [z, y, x]

            # increment batch counter
            batch_counter += 1

            # perform segmentation when a full batch is filled
            if batch_counter == batch_size:

                # convert to tensors
                inputs = torch.FloatTensor(batch).cuda()

                # forward prop
                outputs = self.get_cam(inputs, idx=idx, rescale=rescale, normalize=normalize)
                if input_shape[0] != outputs.size(2) or input_shape[1] != outputs.size(3):
                    outputs = interp(outputs)
                # outputs = F.softmax(outputs, dim=1)

                # cumulate segmentation volume
                for b in range(batch_size):
                    (z_b, y_b, x_b) = positions[b, :]
                    # take into account the gaussian filtering
                    if is2d:
                        seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                            np.multiply(g_window, outputs.data.cpu().numpy()[b, 0:1, :, :])
                        counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
                    else:
                        seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                            np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
                        counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

                # reset batch counter
                batch_counter = 0

        # don't forget last batch
        # convert to tensors
        inputs = torch.FloatTensor(batch).cuda()

        # forward prop
        outputs = self.get_cam(inputs, idx=idx, rescale=rescale, normalize=normalize)
        if input_shape[0] != outputs.size(2) or input_shape[1] != outputs.size(3):
            outputs = interp(outputs)
        # outputs = F.softmax(outputs, dim=1)

        # cumulate segmentation volume
        for b in range(batch_counter):
            (z_b, y_b, x_b) = positions[b, :]
            # take into account the gaussian filtering
            if is2d:
                seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                    np.multiply(g_window, outputs.data.cpu().numpy()[b, 0:1, :, :])
                counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
            else:
                seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                    np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
                counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

        # crop out the symmetric extension and compute segmentation
        segmentation = np.divide(seg_cum[0:counts_cum.shape[0]-2*z_pad, :, :],
                                 counts_cum[0:counts_cum.shape[0] - 2*z_pad, :, :])

        return segmentation

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
