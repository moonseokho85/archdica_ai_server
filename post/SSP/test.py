# System libs
import os
import argparse
from pathlib import Path
from distutils.version import LooseVersion

# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
import cv2

# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

# AWS S3
import boto3


def visualize_result(data, pred, cfg):
    colors = loadmat('data/wall150.mat')['colors']
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]

    # Image.fromarray(pred.astype(np.uint8)).save(os.path.join(cfg.TEST.result, 'pred_' + img_name.replace('.jpg', '.png')))
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


def visualize_mask(data, pred, cfg):
    (img, info) = data

    pred_color = cv2.cvtColor(np.uint8(Image.fromarray(pred)), cv2.COLOR_GRAY2RGB)
    # print('pred_color.shape :', pred_color.shape)
    # print('img.shape :', img.shape)
    im_vis = np.concatenate((img, pred_color), axis=1)
    # print("type of im_vis: ", type(im_vis))

    img_name = info.split('/')[-1]
    # print("img_name: ", img_name)
    # print("cfg.TEST.result: ", cfg.TEST.result)

    # Image.fromarray(pred.astype(np.uint8)).save(os.path.join(cfg.TEST.result, 'pred_' + img_name.replace('.jpg', '.png')))
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

    # Upload through S3
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    AWS_DEFAULT_REGION = "ap-northeast-2"

    s3 = boto3.resource('s3')

    if cfg.TEST.result == './post/SSP/test_result/wall/':

        AWS_BUCKET_NAME = "wall-mask"

        (filename, file_extension) = os.path.splitext(img_name)
        if file_extension is not '.png':
            file_extension = '.png'

        img_name = filename + file_extension
        print('img_name: ', img_name)

        filepath = os.path.join(BASE_DIR, 'post/SSP/test_result/wall/') + img_name
        object_name = img_name

        s3.meta.client.upload_file(filepath, AWS_BUCKET_NAME, object_name, ExtraArgs={'ACL': 'public-read'})

        object_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format(AWS_BUCKET_NAME, AWS_DEFAULT_REGION, img_name)
        print("object_url: ", object_url)

    elif cfg.TEST.result == './post/SSP/test_result/floor/':

        AWS_BUCKET_NAME = "floor-mask"

        (filename, file_extension) = os.path.splitext(img_name)
        if file_extension is not '.png':
            file_extension = '.png'

        img_name = filename + file_extension
        print('img_name: ', img_name)

        filepath = os.path.join(BASE_DIR, 'post/SSP/test_result/floor/') + img_name
        object_name = img_name

        s3.meta.client.upload_file(filepath, AWS_BUCKET_NAME, object_name, ExtraArgs={'ACL': 'public-read'})

        object_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format(AWS_BUCKET_NAME, AWS_DEFAULT_REGION, img_name)
        print("object_url: ", object_url)

    else:
        return


def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:

        # process data
        batch_data = batch_data[0]
        print("batch_data: ", batch_data)
        print("type of batch_data: ", type(batch_data))

        segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            # scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = torch.zeros(1, 2, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, object_index=cfg.MODEL.object_index, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            # _, pred = torch.max(scores, dim=1)
            pred = scores[:, [0], :, :].squeeze(1)
            # print('pred.shape :', pred.shape)
            # break
            pred = as_numpy(pred.squeeze(0).cpu())
            pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255

        # visualization
        # visualize_result(
        #     (batch_data['img_ori'], batch_data['info']),
        #     pred,
        #     cfg
        # )

        visualize_mask(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg)

        pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    unet = ModelBuilder.build_unet(n_channels=5,
                                   n_classes=2,
                                   bilinear=True,
                                   weights=cfg.MODEL.weights_unet)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, unet, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, gpu)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_unet = os.path.join(
        cfg.DIR, 'unet_' + cfg.TEST.checkpoint)

    # print("cfg.MODEL.weights_encoder: ", cfg.MODEL.weights_encoder)
    # print("cfg.MODEL.weights_decoder: ", cfg.MODEL.weights_decoder)
    # print("cfg.MODEL.weights_unet: ", cfg.MODEL.weights_unet)

    # assert os.path.exists(cfg.MODEL.weights_encoder) and os.path.exists(
    #     cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    assert os.path.exists(cfg.MODEL.weights_encoder), "checkpoint of encoder does not exist!"
    assert os.path.exists(cfg.MODEL.weights_decoder), "checkpoint of decoder does not exist!"

    # generate testing image list
    if os.path.isdir(args.imgs):
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
