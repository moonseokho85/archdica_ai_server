from PIL import Image
import gc
import numpy as np
import torch
from timm.data import *
from timm.utils import *
import timm
from pprint import pprint
import torchvision.transforms as trans


def feature_extractor(model_name, imgs_path):
    def imresize(im, size, interp='bilinear'):
        if interp == 'nearest':
            resample = Image.NEAREST
        elif interp == 'bilinear':
            resample = Image.BILINEAR
        elif interp == 'bicubic':
            resample = Image.BICUBIC
        else:
            raise Exception('resample method undefined!')
        return im.resize(size, resample)

    def img_transform(img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = norm(torch.from_numpy(img.copy()))
        return img

    model = timm.create_model(model_name, features_only=True, pretrained=True)
    model = model.cuda()
    model_args = dict(model_name=model_name)

    data_config = timm.data.resolve_data_config(model_args, model=model, verbose=True)
    size = data_config['input_size'][1:]

    norm = trans.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    features = []

    for i, a in enumerate(imgs_path):
        path = a
        img = Image.open(path)

        with torch.no_grad():
            a = imresize(img, size, interp=data_config['interpolation'])
            a = img_transform(a)
            a = a.cuda().unsqueeze(0)
            a = model(a)
        features.append([a, path])
        gc.collect()

    return features


model_name = 'tf_mobilenetv3_large_minimal_100'

test_path = './refer_data/one/'
refer_path = './refer_data/search_lst/'

t = os.listdir(test_path)
for i in range(len(t)):
  t[i] =  test_path + t[i]

r = os.listdir(refer_path)
for i in range(len(r)):
  r[i] =  refer_path + r[i]

target = feature_extractor(model_name, t)
predict = feature_extractor(model_name, r)


def search_(target, refers):
    '''
    target = [[features, filename]]
    refers = [[features, filename] * n]
    '''
    loss_fn = torch.nn.MSELoss(reduction='sum')
    score_dic = dict()

    for i, t in enumerate(target):
        scores = []
        score_dic[t[1]] = []
        for i, p in enumerate(refers):
            score = 0

            for loss_x, loss_y in zip(t[0], p[0]):
                loss = loss_fn(loss_x, loss_y)
                loss = loss.detach().cpu().numpy()
                score += loss

            score = score / len(p[0])
            score_dic[t[1]].append([score, p[1]])
            # scores.append([score, p[1], t[1]])

    save_str = ''
    rank_sum = 0
    for key, val in score_dic.items():
        val = sorted(val, key=lambda x: x[0])
        val_ = [x[1].split('/')[-1] for x in val]
        file_name = key.split('/')[-1]

    return file_name, val_


input_name, similarity_rank = search_(target, predict)
