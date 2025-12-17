import argparse
import os
import random
import glob
import json
import os
import sys
import warnings
from doctest import debug
from pathlib import Path

import numpy as np
import math
import cv2
import torch
import torch.nn.functional as F
import time
from PIL import Image

import tqdm
from matplotlib import cm


def common_args(parser):
    return parser


def keymask_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Keymask Identification')

    # Datasets
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--manualSeed', type=int, default=777, help='manual seed')

    # Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batchSize', default=1, type=int,
                        help='batchSize')

    parser.add_argument('--video-base-path', default='/mnt/data/datasets/DAVIS/JPEGImages/480p',
                        type=str, help='Base path for videos')
    parser.add_argument('--mask-base-path', default='/mnt/data/outputs/DAVIS/cuts3d/pseudo_annotations',
                        type=str, help='Base path for masks')

    parser.add_argument('--save-path', default='/mnt/data/outputs/cotracker/segmentation_masks/DAVIS/all/', type=str)
    parser.add_argument('--video-output-dir', default='/mnt/data/outputs/cotracker/videos', type=str)
    parser.add_argument('--visibility-maps-output-base', default='/mnt/data/outputs/cotracker/visibility_maps', type=str)
    parser.add_argument('--visibility-clusters-output-base', default='/mnt/data/outputs/cotracker/visibility_clusters', type=str)
    parser.add_argument('--annotation-output-path', default='/mnt/data/outputs/cotracker/annotations/DAVIS/all/',
                        type=str)

    parser.add_argument('--visibility-threshold', default=0.3, type=float, help='Threshold for visibility grouping')
    parser.add_argument('--matching-threshold', default=0.5, type=float, help='Threshold for proxy propagate-and-match')

    parser.add_argument('--job-id', default=0, type=int, help='Job ID for distributed training')
    parser.add_argument('--videos-per-job', default=-1, type=int, help='Number of videos to process per job')

    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode')



    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using GPU', args.gpu_id)
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # Set seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    return args

def test_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Label Propagation')

    # Datasets
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--manualSeed', type=int, default=777, help='manual seed')

    # Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batchSize', default=1, type=int,
                        help='batchSize')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='temperature')
    parser.add_argument('--topk', default=10, type=int,
                        help='k for kNN')
    parser.add_argument('--radius', default=12, type=float,
                        help='spatial radius to consider neighbors from')
    parser.add_argument('--videoLen', default=20, type=int,
                        help='number of context frames')

    parser.add_argument('--cropSize', default=320, type=int,
                        help='resizing of test image, -1 for native size')
    parser.add_argument('--feat-bs', default=2, type=int,
                        help='batch size for computing features')

    parser.add_argument('--filelist', default='', type=str)
    parser.add_argument('--data-dir', default='/mnt/data/datasets/ytvis2021/valid', type=str)
    parser.add_argument('--label-dir', default='/mnt/data/outputs/ytvis2021/cuts3d/valid/pseudo_annotations', type=str)
    parser.add_argument('--save-dir', default='/mnt/data/outputs/cotracker/segmentation_masks/ytvis2021/valid/',
                        type=str)
    parser.add_argument('--keyframeslist', default='', type=str)
    parser.add_argument("--visibility-file-dir", default='', type=str)
    parser.add_argument('--save-path', default='./results', type=str)

    parser.add_argument('--visdom', default=False, action='store_true')
    parser.add_argument('--visdom-server', default='localhost', type=str)

    # Model Details
    parser.add_argument('--model-type', default='scratch', type=str)
    parser.add_argument('--vit-arch', default='small', type=str, help='small | base')
    parser.add_argument('--head-depth', default=-1, type=int,
                        help='depth of mlp applied after encoder (0 = linear)')

    parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')
    parser.add_argument('--no-l2', default=False, action='store_true', help='')

    parser.add_argument('--long-mem', default=[0], type=int, nargs='*', help='')
    parser.add_argument('--texture', default=False, action='store_true', help='')
    parser.add_argument('--round', default=False, action='store_true', help='')

    parser.add_argument('--norm_mask', default=False, action='store_true', help='')
    parser.add_argument('--finetune', default=0, type=int, help='')
    parser.add_argument('--pca-vis', default=False, action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using GPU', args.gpu_id)
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # Set seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    return args


def train_args():
    parser = argparse.ArgumentParser(description='Video Walk Training')

    parser.add_argument('--data-path', default='/data/ajabri/kinetics/',
                        help='/home/ajabri/data/places365_standard/train/ | /data/ajabri/kinetics/')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--clip-len', default=8, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--clips-per-video', default=5, type=int, metavar='N',
                        help='maximum number of clips per video to consider')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--steps-per-epoch', default=1e10, type=int, help='max number of batches per epoch')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30, 40], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.3, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='auto', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--partial-reload', default='',
                        help='reload net from checkpoint, ignoring keys that are not in current model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument("--cache-dataset", dest="cache_dataset",
                        help="Cache the datasets for quicker initialization. It also serializes the transforms",
                        action="store_true", )
    parser.add_argument("--data-parallel", dest="data_parallel", help="", action="store_true", )
    parser.add_argument("--fast-test", dest="fast_test", help="", action="store_true", )

    parser.add_argument('--name', default='', type=str, help='')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate on A')
    parser.add_argument('--zero-diagonal', help='always zero diagonal of A', action="store_true", )
    parser.add_argument('--flip', default=False, help='flip transitions (bug)', action="store_true", )

    parser.add_argument('--frame-aug', default='', type=str,
                        help='grid or none')
    parser.add_argument('--frame-transforms', default='crop', type=str,
                        help='combine, ex: crop, cj, flip')

    parser.add_argument('--frame-skip', default=8, type=int, help='kinetics: fps | others: skip between frames')
    parser.add_argument('--img-size', default=256, type=int)
    parser.add_argument('--patch-size', default=[64, 64, 3], type=int, nargs="+")

    parser.add_argument('--port', default=8095, type=int, help='visdom port')
    parser.add_argument('--server', default='localhost', type=str, help='visdom server')

    parser.add_argument('--model-type', default='scratch', type=str, help='scratch | imagenet | moco')
    parser.add_argument('--vit-arch', default='small', type=str, help='small | base')
    parser.add_argument('--optim', default='adam', type=str, help='adam | sgd')

    parser.add_argument('--temp', default=0.07,
                        type=float, help='softmax temperature when computing affinity')
    parser.add_argument('--featdrop', default=0.0, type=float, help='"regular" dropout on features')
    parser.add_argument('--restrict', default=-1, type=int, help='restrict attention')
    parser.add_argument('--head-depth', default=0, type=int, help='depth of head mlp; 0 is linear')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize with wandb and visdom')
    parser.add_argument('--remove-layers', default=[], help='layer[1-4]')

    # sinkhorn-knopp ideas (experimental)
    parser.add_argument('--sk-align', default=False, action='store_true',
                        help='use sinkhorn-knopp to align matches between frames')
    parser.add_argument('--sk-targets', default=False, action='store_true',
                        help='use sinkhorn-knopp to obtain targets, by taking the argmax')

    args = parser.parse_args()

    if args.fast_test:
        args.batch_size = 1
        args.workers = 0
        args.data_parallel = False

    if args.output_dir == 'auto':
        keys = {
            'dropout': 'drop', 'clip_len': 'len', 'frame_transforms': 'ftrans', 'frame_aug': 'faug',
            'optim': 'optim', 'temp': 'temp', 'featdrop': 'fdrop', 'lr': 'lr', 'head_depth': 'mlp'
        }
        name = '-'.join(["%s%s" % (keys[k], getattr(args, k) if not isinstance(getattr(args, k), list) else '-'.join(
            [str(s) for s in getattr(args, k)])) for k in keys])
        args.output_dir = "/mnt/hdd/leon/outputs/crw/checkpoints/%s_%s_%s/" % (args.model_type, args.name, name)

        import datetime
        dt = datetime.datetime.today()
        args.name = "%s-%s-%s_%s" % (str(dt.month), str(dt.day), args.name, name)

    os.makedirs(args.output_dir, exist_ok=True)

    return args





def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize(img, (owidth, oheight))
    img = im_to_torch(img)
    return img

def safe_exists(p):
    try:
        return Path(p).exists()
    except OSError as e:
        # log it, skip the file, or handle as appropriate
        print(f"Warning: I/O error checking {p}: {e}")
        return False

def load_image_robust(path, max_retries=3, backoff=0.1):
    """
    Try to load an image using OpenCV, with retries, and fallback to PIL if needed.
    Returns BGR image like cv2.imread, or raises an IOError.
    """
    path = Path(path)
    if not safe_exists(path):
        warnings.warn(f"File does not exist or the loading gracefully failed for: {path!s}")
        return None

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # Quick sanity check: ensure we can open the file bytes
            with open(path, "rb") as f:
                header = f.read(16)
                if len(header) == 0:
                    raise IOError("File appears empty or unreadable")

            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                return img  # success

            # Fallback: try PIL
            pil_img = Image.open(path)
            pil_img.verify()  # verify integrity
            pil_img = Image.open(path)  # reopen after verify
            # Convert PIL to OpenCV BGR
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            last_exc = e
            print("Attempt %d to load %s failed: %s", attempt, path, e)
            time.sleep(backoff * attempt)  # simple exponential backoff

    warnings.warn(f"Failed to load image {path!s} after {max_retries} attempts. Last error: {last_exc}")
    return None
    #raise IOError(f"Failed to load image {path!s} after {max_retries} attempts. Last error: {last_exc}")



def load_image(img_path):
    # H x W x C => C x H x W
    #img = cv2.imread(img_path)
    img = load_image_robust(img_path)
    if img is None:
        return None

    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = img.copy()
    return im_to_torch(img)


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


######################################################################
def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None


def make_lbl_set(lbls):
    lbl_set = [np.zeros(3).astype(np.uint8)]
    count_lbls = [0]

    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)
    # print("basis flat_lbls_0", flat_lbls_0.shape)
    # print(np.min(flat_lbls_0), np.max(flat_lbls_0))
    # print("Created lblset with shape", lbl_set.shape)
    """
    basis flat_lbls_0 (409920, 3)
    Created lblset with shape (2, 3)

    Created lblset with shape (2, 3)
    Created lblset with shape (3, 3)
    Created lblset with shape (3, 3)

    Created lblset with shape (3688, 3)
    Created lblset with shape (9696, 3)
    Created lblset with shape (2116, 3)
    """

    return lbl_set


def resize_torch(one_hot, rsz_h, rsz_w):
    # Assume one_hot is a numpy array of shape (480, 854, 3688)
    # Convert to float32 if needed and then to a torch tensor.
    one_hot = one_hot.astype(np.float32)
    one_hot_tensor = torch.from_numpy(one_hot)  # shape: (480, 854, 3688)

    # Permute the dimensions to (channels, height, width)
    one_hot_tensor = one_hot_tensor.permute(2, 0, 1)  # shape: (3688, 480, 854)

    # Add a batch dimension: (1, channels, height, width)
    one_hot_tensor = one_hot_tensor.unsqueeze(0)

    # Define the target size. Note: the size is specified as (target_height, target_width)
    target_size = (rsz_h, rsz_w)  # replace rsz_h and rsz_w with your desired dimensions

    # Resize using bilinear interpolation.
    # align_corners=False generally corresponds to OpenCV's INTER_LINEAR.
    resized_tensor = F.interpolate(one_hot_tensor, size=target_size, mode='bilinear', align_corners=False)

    # Optionally, if you want to convert back to a numpy array with shape (target_height, target_width, 3688):
    # First, remove the batch dimension and then permute back to (height, width, channels)
    resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)
    resized = resized_tensor.cpu().numpy()

    return resized


def texturize(onehot):
    flat_onehot = onehot.reshape(-1, onehot.shape[-1])
    lbl_set = np.unique(flat_onehot, axis=0)

    count_lbls = [np.all(flat_onehot == ll, axis=-1).sum() for ll in lbl_set]
    object_id = np.argsort(count_lbls)[::-1][1]

    hidxs = []
    for h in range(onehot.shape[0]):
        appears = np.any(onehot[h, :, 1:] == 1)
        if appears:
            hidxs.append(h)

    nstripes = min(10, len(hidxs))

    out = np.zeros((*onehot.shape[:2], nstripes + 1))
    out[:, :, 0] = 1

    for i, h in enumerate(hidxs):
        cidx = int(i // (len(hidxs) / nstripes))
        w = np.any(onehot[h, :, 1:] == 1, axis=-1)
        out[h][w] = 0
        out[h][w, cidx + 1] = 1

    return out


class VOSDataset(torch.utils.data.Dataset):
    def __init__(self, args):

        self.filelist = args.filelist
        if hasattr(args, 'keyframeslist'):
            if len(args.keyframeslist) > 0:
                self.keyframeslist = args.keyframeslist

                f = open(self.keyframeslist, 'r')
                self.keyframes = []
                for line in f:
                    rows = line.split()
                    self.keyframes.append(int(rows[0]))

                f.close()
            else:
                self.keyframeslist = None
        else:
            self.keyframeslist = None

        self.imgSize = args.imgSize
        print("self.imgSize", self.imgSize)
        self.videoLen = args.videoLen
        self.mapScale = args.mapScale

        self.texture = args.texture
        self.round = args.round
        self.use_lab = getattr(args, 'use_lab', False)

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            lblfile = rows[1]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()

    def get_onehot_lbl(self, lbl_path):
        name = '/' + '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None

    def make_paths(self, folder_path, label_path):
        I, L = os.listdir(folder_path), os.listdir(label_path)
        L = [ll for ll in L if 'npy' not in ll]

        frame_num = len(I) + self.videoLen
        I.sort(key=lambda x: int(x.split('.')[0]))
        L.sort(key=lambda x: int(x.split('.')[0]))

        I_out, L_out = [], []

        for i in range(frame_num):
            i = max(0, i - self.videoLen)
            img_path = "%s/%s" % (folder_path, I[i])
            lbl_path = "%s/%s" % (label_path, L[i])

            I_out.append(img_path)
            L_out.append(lbl_path)

        return I_out, L_out

    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        imgs_orig = []
        lbls = []
        lbls_onehot = []
        patches = []
        target_imgs = []

        frame_num = len(os.listdir(folder_path)) + self.videoLen

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img_paths, lbl_paths = self.make_paths(folder_path, label_path)

        t000 = time.time()

        for i in range(frame_num):
            t00 = time.time()

            img_path, lbl_path = img_paths[i], lbl_paths[i]
            img = load_image(img_path)  # CxHxW
            lblimg = cv2.imread(lbl_path)

            ht, wd = img.size(1), img.size(2)
            if self.imgSize > 0:
                newh, neww = ht, wd

                if ht <= wd:
                    ratio = 1.0  # float(wd) / float(ht)
                    # width, height
                    img = resize(img, int(self.imgSize * ratio), self.imgSize)
                    newh = self.imgSize
                    neww = int(self.imgSize * ratio)
                else:
                    ratio = 1.0  # float(ht) / float(wd)
                    # width, height
                    img = resize(img, self.imgSize, int(self.imgSize * ratio))
                    newh = int(self.imgSize * ratio)
                    neww = self.imgSize

                lblimg = cv2.resize(lblimg, (newh, neww), cv2.INTER_NEAREST)

            vit_ht = int(math.ceil(ht / 8) * 8)
            vit_wd = int(math.ceil(wd / 8) * 8)

            if int(ht) != vit_ht or int(wd) != vit_wd:
                img = resize(img, vit_wd, vit_ht)
                newh, neww = ht, wd

                lblimg = cv2.resize(lblimg, (neww, newh), cv2.INTER_NEAREST)

            img_orig = img.clone()

            if self.use_lab:
                img = im_to_numpy(img)
                img = (img * 255).astype(np.uint8)[:, :, ::-1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img = im_to_torch(img)
                img = color_normalize(img, [128, 128, 128], [128, 128, 128])
                img = torch.stack([img[0]] * 3)
            else:
                img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)
            lbls.append(lblimg.copy())

        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths)

        if self.keyframeslist is not None:
            meta['keyframe'] = self.keyframes[index]
        else:
            meta['keyframe'] = -1

        ########################################################
        # Load reshaped label information (load cached versions if possible)
        lbls = np.stack(lbls)
        prefix = '/' + '/'.join(lbl_paths[0].split('.')[:-1])

        # Get lblset
        lblset_path = "%s_%s.npy" % (prefix, 'lblset')
        lblset = make_lbl_set(lbls)

        if np.all((lblset[1:] - lblset[:-1]) == 1):
            lblset = lblset[:, 0:1]

        # import pdb; pdb.set_trace()
        # lblset = try_np_load(lblset_path)
        # if lblset is None or True:
        #     print('making label set', lblset_path)
        #     lblset = make_lbl_set(lbls)
        #     np.save(lblset_path, lblset)

        onehots = []
        resizes = []

        rsz_h, rsz_w = math.ceil(img.size(1) / self.mapScale[0]), math.ceil(img.size(2) / self.mapScale[1])

        for i, p in enumerate(lbl_paths):
            prefix = '/' + '/'.join(p.split('.')[:-1])
            # print(prefix)
            oh_path = "%s_%s.npy" % (prefix, 'onehot')
            rz_path = "%s_%s.npy" % (prefix, 'size%sx%s' % (rsz_h, rsz_w))

            onehot = try_np_load(oh_path)
            if onehot is None:
                print('computing onehot lbl for', oh_path)
                onehot = np.stack([np.all(lbls[i] == ll, axis=-1) for ll in lblset], axis=-1)
                np.save(oh_path, onehot)

            resized = try_np_load(rz_path)
            if resized is None:
                print('computing resized lbl for', rz_path)
                print(onehot.shape)
                # print(type(onehot)) np array
                print(onehot.dtype)  # bool
                # try:
                resized = cv2.resize(np.float32(onehot), (rsz_w, rsz_h), cv2.INTER_LINEAR)
                # resized = resize_torch(onehot, rsz_h, rsz_w)
                np.save(rz_path, resized)

                # sys.exit(0)

            if self.texture:
                texturized = texturize(resized)
                resizes.append(texturized)
                lblset = np.array([[0, 0, 0]] + [cm.Paired(i)[:3] for i in range(texturized.shape[-1])]) * 255.0
                break
            else:
                resizes.append(resized)
                onehots.append(onehot)

        if self.texture:
            resizes = resizes * self.videoLen
            for _ in range(len(lbl_paths) - self.videoLen):
                resizes.append(np.zeros(resizes[0].shape))
            onehots = resizes

        ########################################################

        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        lbls_tensor = torch.from_numpy(np.stack(lbls))
        lbls_resize = np.stack(resizes)

        assert lbls_resize.shape[0] == len(meta['lbl_paths'])

        return imgs, imgs_orig, lbls_resize, lbls_tensor, lblset, meta

    def __len__(self):
        return len(self.jpgfiles)


def convert_lblimg_to_maskid(lblimg: np.ndarray) -> torch.Tensor:
    H, W, _ = lblimg.shape
    # 3) find all unique colors in this frame
    pixels = lblimg.reshape(-1, 3)
    uniq = np.unique(pixels, axis=0)
    # filter out black
    colors = [tuple(c) for c in uniq if not np.all(c == 0)]

    # 4) build a color→ID map (black→0, others 1..)
    color2id = {(0, 0, 0): 0}
    for idx, col in enumerate(sorted(colors), start=1):
        color2id[col] = idx

    # 5) create the ID map for this frame
    id_map = np.zeros((H, W), dtype=np.int64)
    for col, idx in color2id.items():
        if idx == 0:
            continue
        mask = np.all(lblimg == col, axis=2)
        id_map[mask] = idx

    label_map = id_map[..., None]

    return label_map


def load_masks(mask_folder: str) -> torch.Tensor:
    """
    Load an ordered sequence of multi‐color PNG masks from a folder and
    convert them into a single mask‐ID tensor.

    Args:
        mask_folder: Path to a directory containing one .png mask per frame.
                     Other files are ignored.

    Returns:
        masks: torch.Tensor of shape (T, H, W, 1), dtype torch.long,
               where each pixel is 0 for background or 1..N for each color‐mask.
    """
    # 1) collect all .png paths
    paths = sorted(glob.glob(os.path.join(mask_folder, '*.png')))
    if not paths:
        raise ValueError(f"No .png masks found in {mask_folder!r}")

    id_maps = []
    for p in paths:
        # 2) read as RGB
        bgr = load_image_robust(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = rgb.shape

        # 3) find all unique colors in this frame
        pixels = rgb.reshape(-1, 3)
        uniq = np.unique(pixels, axis=0)
        # filter out black
        colors = [tuple(c) for c in uniq if not np.all(c == 0)]

        # 4) build a color→ID map (black→0, others 1..)
        color2id = {(0, 0, 0): 0}
        for idx, col in enumerate(sorted(colors), start=1):
            color2id[col] = idx

        # 5) create the ID map for this frame
        id_map = np.zeros((H, W), dtype=np.int64)
        for col, idx in color2id.items():
            if idx == 0:
                continue
            mask = np.all(rgb == col, axis=2)
            id_map[mask] = idx

        id_maps.append(id_map[..., None])  # (H, W, 1)

    if not id_maps:
        raise RuntimeError("No valid mask images could be read.")

    # 6) stack into (T, H, W, 1) and convert to tensor
    masks_np = np.stack(id_maps, axis=0)
    return torch.from_numpy(masks_np)  # dtype=torch.long

def make_paths(folder_path, label_path, dataset_name='DAVIS'):
    I_pre, L = os.listdir(folder_path), os.listdir(label_path)
    L = [ll for ll in L if 'npy' not in ll]

    # Make sure all files in I are images and no folders
    I = [i for i in I_pre if i.endswith(('.jpg', '.png', '.jpeg'))]

    frame_num = len(I)

    if dataset_name == "SA-V":
        I.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        L.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    else:
        I.sort(key=lambda x: int(x.split('.')[0])) if dataset_name != 'ovis' else I.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        L.sort(key=lambda x: int(x.split('.')[0])) if dataset_name != 'ovis' else I.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    I_out, L_out = [], []

    for i in range(frame_num):
        img_path = "%s/%s" % (folder_path, I[i])
        lbl_path = "%s/%s" % (label_path, L[i])

        I_out.append(img_path)
        L_out.append(lbl_path)

    return I_out, L_out

def load_frames_and_masks(video_path, label_path, visibility_data, dataset_name='DAVIS'):
    imgs = []
    imgs_orig = []
    lbls = []

    frame_num = len(os.listdir(video_path))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_paths, lbl_paths = make_paths(video_path, label_path, dataset_name)

    t000 = time.time()

    for i in tqdm.tqdm(range(frame_num), desc="Loading Masks.."):
        t00 = time.time()

        img_path, lbl_path = img_paths[i], lbl_paths[i]
        img = load_image(img_path)  # CxHxW
        #lblimg = cv2.imread(lbl_path)
        lblimg = load_image_robust(lbl_path)

        if img is None or lblimg is None:
            return None, None, None, None

        ht, wd = img.size(1), img.size(2)

        needs_to_fit_vit = False
        if needs_to_fit_vit:
            vit_ht = int(math.ceil(ht / 8) * 8)
            vit_wd = int(math.ceil(wd / 8) * 8)

            if int(ht) != vit_ht or int(wd) != vit_wd:
                img = resize(img, vit_wd, vit_ht)
                newh, neww = ht, wd

                lblimg = cv2.resize(lblimg, (neww, newh), cv2.INTER_NEAREST)

        lblimg = convert_lblimg_to_maskid(lblimg)

        img_orig = img.clone()

        img = color_normalize(img, mean, std)

        imgs_orig.append(img_orig)
        imgs.append(img)
        lbls.append(lblimg.copy())

        # Load visibility file

    # Meta info
    meta = dict(video_path=video_path, img_paths=img_paths, lbl_paths=lbl_paths, visibility=visibility_data)

    imgs = torch.stack(imgs)
    imgs_orig = torch.stack(imgs_orig)
    # lbls_tensor = torch.from_numpy(np.stack(lbls)).permute(0, 3, 1, 2)
    lbls_tensor = load_masks(label_path).permute(0, 3, 1, 2)  # shape: (T, H, W, 1) -> (T, 1, H, W)

    # Resize lbls tensor so that last two dims match imgs tensor
    lbls_tensor = F.interpolate(lbls_tensor.float(), size=(imgs.size(2), imgs.size(3)), mode='nearest').long()

    #print(imgs.shape, imgs_orig.shape, lbls_tensor.shape)

    return imgs, imgs_orig, lbls_tensor, meta


class PropagateAndMatchDAVIS(torch.utils.data.Dataset):
    def __init__(self, args):
        print("Constructing PropagateAndMatch DAVIS dataset...")
        self.filelist = args.filelist
        self.visibility_file_dir = args.visibility_file_dir
        self.imgSize = args.imgSize
        self.videoLen = args.videoLen
        self.mapScale = args.mapScale

        self.texture = args.texture
        self.round = args.round
        self.use_lab = getattr(args, 'use_lab', False)

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []
        self.visibility_files = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            lblfile = rows[1]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

            # Get the folder name by splitting at / and then taking -1 entry
            video_name = jpgfile.split('/')[-1]
            visibility_file = os.path.join(self.visibility_file_dir, f"{video_name}.json")
            self.visibility_files.append(visibility_file)

        f.close()

    def get_onehot_lbl(self, lbl_path):
        name = '/' + '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None

    def make_paths(self, folder_path, label_path):
        I, L = os.listdir(folder_path), os.listdir(label_path)
        L = [ll for ll in L if 'npy' not in ll]

        frame_num = len(I) + self.videoLen
        I.sort(key=lambda x: int(x.split('.')[0]))
        L.sort(key=lambda x: int(x.split('.')[0]))

        I_out, L_out = [], []

        for i in range(frame_num):
            i = max(0, i - self.videoLen)
            img_path = "%s/%s" % (folder_path, I[i])
            lbl_path = "%s/%s" % (label_path, L[i])

            I_out.append(img_path)
            L_out.append(lbl_path)

        return I_out, L_out


    def __getitem__(self, index):
        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]
        vibility_file_path = self.visibility_files[index]

        imgs = []
        imgs_orig = []
        lbls = []
        lbls_onehot = []
        patches = []
        target_imgs = []

        frame_num = len(os.listdir(folder_path)) + self.videoLen

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img_paths, lbl_paths = self.make_paths(folder_path, label_path)

        t000 = time.time()

        for i in range(frame_num):
            t00 = time.time()

            img_path, lbl_path = img_paths[i], lbl_paths[i]
            img = load_image(img_path)  # CxHxW
            lblimg = cv2.imread(lbl_path)

            ht, wd = img.size(1), img.size(2)
            if self.imgSize > 0:
                newh, neww = ht, wd

                if ht <= wd:
                    ratio = 1.0  # float(wd) / float(ht)
                    # width, height
                    img = resize(img, int(self.imgSize * ratio), self.imgSize)
                    newh = self.imgSize
                    neww = int(self.imgSize * ratio)
                else:
                    ratio = 1.0  # float(ht) / float(wd)
                    # width, height
                    img = resize(img, self.imgSize, int(self.imgSize * ratio))
                    newh = int(self.imgSize * ratio)
                    neww = self.imgSize

                lblimg = cv2.resize(lblimg, (newh, neww), cv2.INTER_NEAREST)

            vit_ht = int(math.ceil(ht / 8) * 8)
            vit_wd = int(math.ceil(wd / 8) * 8)

            if int(ht) != vit_ht or int(wd) != vit_wd:
                img = resize(img, vit_wd, vit_ht)
                newh, neww = ht, wd

                lblimg = cv2.resize(lblimg, (neww, newh), cv2.INTER_NEAREST)

            lblimg = convert_lblimg_to_maskid(lblimg)

            img_orig = img.clone()

            if self.use_lab:
                img = im_to_numpy(img)
                img = (img * 255).astype(np.uint8)[:, :, ::-1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img = im_to_torch(img)
                img = color_normalize(img, [128, 128, 128], [128, 128, 128])
                img = torch.stack([img[0]] * 3)
            else:
                img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)
            lbls.append(lblimg.copy())

            # Load visibility file
        with open(vibility_file_path, 'r') as vf:
            visibility_data = json.load(vf)

        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths, visibility=visibility_data)

        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        # lbls_tensor = torch.from_numpy(np.stack(lbls)).permute(0, 3, 1, 2)
        lbls_tensor = load_masks(label_path).permute(0, 3, 1, 2)  # shape: (T, H, W, 1) -> (T, 1, H, W)

        # Resize lbls tensor so that last two dims match imgs tensor
        lbls_tensor = F.interpolate(lbls_tensor.float(), size=(imgs.size(2), imgs.size(3)), mode='nearest').long()

        print(imgs.shape, imgs_orig.shape, lbls_tensor.shape)

        return imgs, imgs_orig, lbls_tensor, meta

    def __len__(self):
        return len(self.jpgfiles)


class PropagateAndMatchYTVIS(torch.utils.data.Dataset):
    def __init__(self, args):
        print("Constructing PropagateAndMatch YTVIS dataset...")
        self.visibility_file_dir = args.visibility_file_dir
        self.imgSize = args.imgSize
        self.videoLen = args.videoLen
        self.mapScale = args.mapScale

        self.texture = args.texture
        self.round = args.round
        self.use_lab = getattr(args, 'use_lab', False)

        self.jpgfiles = [os.path.join(args.data_dir, path) for path in os.listdir(args.data_dir)]
        self.lblfiles = [os.path.join(args.label_dir, path) for path in os.listdir(args.label_dir)]

        self.visibility_files = []
        self.jpgfiles_filtered = []
        self.lblfiles_filtered = []

        for jpgfile in self.jpgfiles:
            # Get the folder name by splitting at / and then taking -1 entry
            video_name = jpgfile.split('/')[-1]
            visibility_file = os.path.join(self.visibility_file_dir, f"{video_name}.json")

            if not os.path.exists(visibility_file):
                print(f"Visibility file not found: {visibility_file}")
                continue

            self.visibility_files.append(visibility_file)
            self.jpgfiles_filtered.append(jpgfile)
            self.lblfiles_filtered.append(os.path.join(args.label_dir, video_name))

        self.jpgfiles = self.jpgfiles_filtered
        self.lblfiles = self.lblfiles_filtered

    def get_onehot_lbl(self, lbl_path):
        name = '/' + '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None

    def make_paths(self, folder_path, label_path):
        I, L = os.listdir(folder_path), os.listdir(label_path)
        L = [ll for ll in L if 'npy' not in ll]

        frame_num = len(I) + self.videoLen
        I.sort(key=lambda x: int(x.split('.')[0]))
        L.sort(key=lambda x: int(x.split('.')[0]))

        I_out, L_out = [], []

        for i in range(frame_num):
            i = max(0, i - self.videoLen)
            img_path = "%s/%s" % (folder_path, I[i])
            lbl_path = "%s/%s" % (label_path, L[i])

            I_out.append(img_path)
            L_out.append(lbl_path)

        return I_out, L_out

    def __getitem__(self, index):
        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]
        vibility_file_path = self.visibility_files[index]

        imgs = []
        imgs_orig = []
        lbls = []
        lbls_onehot = []
        patches = []
        target_imgs = []

        frame_num = len(os.listdir(folder_path)) + self.videoLen

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img_paths, lbl_paths = self.make_paths(folder_path, label_path)

        t000 = time.time()

        for i in range(frame_num):
            t00 = time.time()

            img_path, lbl_path = img_paths[i], lbl_paths[i]
            img = load_image(img_path)  # CxHxW
            lblimg = cv2.imread(lbl_path)

            ht, wd = img.size(1), img.size(2)
            if self.imgSize > 0:
                newh, neww = ht, wd

                if ht <= wd:
                    ratio = 1.0  # float(wd) / float(ht)
                    # width, height
                    img = resize(img, int(self.imgSize * ratio), self.imgSize)
                    newh = self.imgSize
                    neww = int(self.imgSize * ratio)
                else:
                    ratio = 1.0  # float(ht) / float(wd)
                    # width, height
                    img = resize(img, self.imgSize, int(self.imgSize * ratio))
                    newh = int(self.imgSize * ratio)
                    neww = self.imgSize

                lblimg = cv2.resize(lblimg, (newh, neww), cv2.INTER_NEAREST)

            vit_ht = int(math.ceil(ht / 8) * 8)
            vit_wd = int(math.ceil(wd / 8) * 8)

            if int(ht) != vit_ht or int(wd) != vit_wd:
                img = resize(img, vit_wd, vit_ht)
                newh, neww = ht, wd

                lblimg = cv2.resize(lblimg, (neww, newh), cv2.INTER_NEAREST)

            lblimg = convert_lblimg_to_maskid(lblimg)

            img_orig = img.clone()

            if self.use_lab:
                img = im_to_numpy(img)
                img = (img * 255).astype(np.uint8)[:, :, ::-1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img = im_to_torch(img)
                img = color_normalize(img, [128, 128, 128], [128, 128, 128])
                img = torch.stack([img[0]] * 3)
            else:
                img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)
            lbls.append(lblimg.copy())

            # Load visibility file
        try:
            with open(vibility_file_path, 'r') as vf:
                visibility_data = json.load(vf)
        except FileNotFoundError:
            print(f"Visibility file not found: {vibility_file_path}")
            visibility_data = {}

        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths, visibility=visibility_data)

        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        # lbls_tensor = torch.from_numpy(np.stack(lbls)).permute(0, 3, 1, 2)
        lbls_tensor = load_masks(label_path).permute(0, 3, 1, 2)  # shape: (T, H, W, 1) -> (T, 1, H, W)

        # Resize lbls tensor so that last two dims match imgs tensor
        lbls_tensor = F.interpolate(lbls_tensor.float(), size=(imgs.size(2), imgs.size(3)), mode='nearest').long()

        print(imgs.shape, imgs_orig.shape, lbls_tensor.shape)

        return imgs, imgs_orig, lbls_tensor, meta

    def __len__(self):
        return len(self.jpgfiles)


