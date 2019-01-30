
"""
    This is a script that tests a U-Net
    Usage:
        python train.py --method 2D --net /path/to/network.pytorch
"""

"""
    Necessary libraries
"""
import numpy as np
import os
import argparse
import datetime

from util.tools import load_net
from util.io import imwrite3D, read_tif
from util.preprocessing import normalize
from util.metrics import accuracy_metrics, jaccard, dice

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--write_dir", help="Writing directory", type=str, default=None)

# data parameters
parser.add_argument("--data", help="Path to the data (should be tif file)", type=str, default=None)
parser.add_argument("--data_labels", help="Path to the data labels (should be tif file)", type=str, default=None)

# network parameters
parser.add_argument("--net", help="Path to the network", type=str, default="checkpoint.pytorch")

# optimization parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="512,512")
parser.add_argument("--batch_size", help="Batch size", type=int, default=1)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]

"""
    Setup writing directory
"""
print('[%s] Setting up write directories' % (datetime.datetime.now()))
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)

"""
    Load and normalize the data
"""
print('[%s] Loading and normalizing the data' % (datetime.datetime.now()))
test_data = read_tif(args.data, dtype='uint8')
mu = np.mean(test_data)
std = np.std(test_data)
test_data = normalize(test_data, mu, std)
if len(test_data.shape)<3:
    test_data = test_data[np.newaxis, ...]

"""
    Load the network
"""
print('[%s] Loading network' % (datetime.datetime.now()))
net = load_net(args.net)

"""
    Segmentation
"""
print('[%s] Starting segmentation' % (datetime.datetime.now()))
segmentation = net.segment_cam(test_data, args.input_size, batch_size=args.batch_size, normalize=False)
m = np.min(segmentation)
M = np.max(segmentation)
segmentation = (segmentation - m) / (M - m)

"""
    Validate the segmentation
"""
if args.data_labels is not None:
    print('[%s] Validating segmentation' % (datetime.datetime.now()))
    test_data_labels = read_tif(args.data_labels, dtype='uint8')
    test_data_labels = normalize(test_data_labels, 0, 255)
    j = jaccard(segmentation, test_data_labels)
    d = dice(segmentation, test_data_labels)
    a, p, r, f = accuracy_metrics(segmentation, test_data_labels)
    print('[%s] RESULTS:' % (datetime.datetime.now()))
    print('[%s]     Jaccard: %f' % (datetime.datetime.now(), j))
    print('[%s]     Dice: %f' % (datetime.datetime.now(), d))
    print('[%s]     Accuracy: %f' % (datetime.datetime.now(), a))
    print('[%s]     Precision: %f' % (datetime.datetime.now(), p))
    print('[%s]     Recall: %f' % (datetime.datetime.now(), r))
    print('[%s]     F-score: %f' % (datetime.datetime.now(), f))

"""
    Write out the results
"""
if args.write_dir is not None:
    print('[%s] Writing the output' % (datetime.datetime.now()))
    imwrite3D(segmentation, args.write_dir, rescale=True)

print('[%s] Finished!' % (datetime.datetime.now()))