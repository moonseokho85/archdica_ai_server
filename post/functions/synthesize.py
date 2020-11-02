import cv2
from PIL import Image
import os
from .layout_estimation_func import refering
import numpy as np
import boto3
from io import BytesIO
import environ

env = environ.Env()
environ.Env.read_env()

def synthesize(room_image_url, object_type):

    object_ = object_type

    # S3 client
    client = boto3.client('s3')

    # S3 key of image
    key = os.path.basename(room_image_url)
    print("key: ", key)

    # Load mask image from S3
    if object_ == "wall":

        bucket = 'wall-mask'

        file_byte_string = client.get_object(Bucket=bucket, Key=key)['Body'].read()
        image = Image.open(BytesIO(file_byte_string))

    elif object_ == "floor":

        bucket = 'floor-mask'

        file_byte_string = client.get_object(Bucket=bucket, Key=key)['Body'].read()
        image = Image.open(BytesIO(file_byte_string))

    else:
        return

    # Load reference image from S3
    bucket = 'archdica-material'

    file_byte_string = client.get_object(Bucket=bucket, Key=key)['Body'].read()
    refer = Image.open(BytesIO(file_byte_string))

    org, mask_img = np.split(np.array(image), 2, axis=1)
    # print(org.shape, mask_img.shape)

    img_size = (mask_img.shape[1], mask_img.shape[0])

    # refer = refer.resize(img_size)
    #             Use refering            #
    #       org, layout_img, refer 's type = ndarray        #
    refer = refer
    # refer = layout_estimation_func.refering(org, layout, refer)

    scale_factor = 6
    print('scale_factor :', scale_factor)

    # refer = Image.open(refer_path)
    # refer = np.asarray(refer)

    import math

    #     refer_size와 img_size가 동일하거나 refer_size가 작은 경우를 고려해야한다.     #
    refer = np.tile(refer, (scale_factor, scale_factor, 1))
    size_ratio = math.floor(min((refer.shape[0] / (org.shape[0] * 1.5)), (refer.shape[1] / (org.shape[1] * 1.5))))
    # refer = Image.fromarray(refer).resize((int(refer.shape[1] / size_ratio), int(refer.shape[0] / size_ratio)))
    refer = Image.fromarray(refer).resize((org.shape[1], org.shape[0]))

    #     refer_image Warping     #
    #          Find Max Height Contour Point       #
    if object_ is not 'wall':

        mask_img_gr = mask_img[:mask_img.shape[0] - 10, :mask_img.shape[1] - 10]
        gray = cv2.cvtColor(mask_img_gr, cv2.COLOR_RGB2GRAY)
        # gray = cv2.cvtColor(mask_img.astype(np.float32), cv2.COLOR_RGB2GRAY) * 255
        ret, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours.shape)

        if object_ == 'floor':
            max_height_point = np.array([[mask_img.shape[0], mask_img.shape[1]]])
            # print(max_height_point)
            for contour in contours:
                # print(contour.shape)
                for point in contour:
                    # print(point[0][1], max_height_point[0][1])
                    if 0 < point[0][1] <= max_height_point[0][1]:
                        # print(max_height_point[0][1])
                        max_height_point = point
                    # print(point)
                    # break
                # ctr = contour.astype(np.int32)
                # cv2.drawContours(mask_img, [ctr], -1, (255, 0, 0), 5)

            #       Warping       #
            pts1 = np.float32([max_height_point[0], [0, gray.shape[1]], [gray.shape[0], gray.shape[1]]])
            pts2 = np.float32([[max_height_point[0][0], 0], [0, gray.shape[1]], [gray.shape[0], gray.shape[1]]])

        elif object_ == 'ceiling':
            max_height_point = np.array([[0, 0]])
            # print(max_height_point)
            for contour in contours:
                # print(contour.shape)
                for point in contour:
                    # print(point[0][1], max_height_point[0][1])
                    if point[0][1] >= max_height_point[0][1]:
                        # print(max_height_point[0][1])
                        max_height_point = point
                    # print(point)
                    # break
                # ctr = contour.astype(np.int32)
                # cv2.drawContours(mask_img, [ctr], -1, (255, 0, 0), 5)

            #       Warping       #
            pts1 = np.float32([max_height_point[0], [0, 0], [gray.shape[0], 0]])
            pts2 = np.float32([[max_height_point[0][0], gray.shape[1]], [0, 0], [gray.shape[0], 0]])

        matrix = cv2.getAffineTransform(pts2, pts1)

        refer = np.asarray(refer)
        refer = cv2.warpAffine(refer, matrix, (refer.shape[1], refer.shape[0]))

    # plt.imshow(refer)
    # plt.show()
    # break
    print('mask_img.shape :', mask_img.shape)
    mask_img = mask_img / 255.

    #     1.wall_mask * reference color image + (1 - wall_mask) * original_image)    #
    syn_ = mask_img * (refer) + (1 - mask_img) * org
    syn = mask_img * syn_ + (1 - mask_img) * org

    # syn = mask_img * (0.5 * np.array(refer) + 0.5 * org) + (1 - mask_img) * org
    # print(syn.max(), syn.min())
    # print(syn.shape)

    #       Brightness Reservation      #
    #     2.1st result's hsv('value') channel = w * wall_mask * reference hsv('value') + (1 - w) * (1 - wall_mask) * 1st hsv('value')
    org_hsv = cv2.cvtColor(np.uint8(org), cv2.COLOR_RGB2HSV)
    refer_hsv = cv2.cvtColor(np.uint8(refer), cv2.COLOR_RGB2HSV)
    syn_hsv = cv2.cvtColor(np.uint8(syn), cv2.COLOR_RGB2HSV)

    hsv_added = cv2.addWeighted(org, 0.7, org_hsv, 0.3, 0)
    kernel_size, low_threshold, high_threshold = 5, 0, 150
    hsv_added = cv2.GaussianBlur(hsv_added, (kernel_size, kernel_size), 0)
    hsv_added = cv2.Canny(hsv_added, low_threshold, high_threshold)
    hsv_added = np.invert(hsv_added)

    # plt.imshow(hsv_added)
    # plt.show()

    org_h, org_s, org_v = cv2.split(org_hsv)
    refer_h, refer_s, refer_v = cv2.split(refer_hsv)
    syn_h, syn_s, syn_v = cv2.split(syn_hsv)

    # mask_img = cv2.cvtColor(np.uint8(mask_img), cv2.COLOR_RGB2GRAY)
    # print(mask_img.shape)
    # print(refer_v.shape)
    # print(syn_v.shape)
    # final_syn_hsv = cv2.merge([syn_h, syn_s, syn_v])
    # plt.imshow(np.hstack((syn_h, syn_s, syn_v)))
    # plt.show()
    # plt.imshow(np.hstack((org_h, org_s, org_v)))

    w = .5
    # print(syn_v.min(), syn_v.max())
    # syn_v = w * mask_img * refer_v + (1 - w) * (1 - mask_img) * org_v
    prev_syn_v = syn_v.copy()
    syn_v_im = w * syn_v + (1 - w) * org_v
    syn_v = w * syn_v_im + (1 - w) * org_v
    # syn_v = mask_img * syn_v * k + (1 - mask_img) * org_v
    syn_v = syn_v.astype(np.uint8)
    # print(syn_v.min(), syn_v.max())

    # prev_syn_h = syn_h.copy()
    # syn_h = w * syn_h + (1 - w) * org_h
    # syn_h = syn_h.astype(np.uint8)

    # syn_s = w * syn_s + (1 - w) * org_s
    # syn_s = syn_s.astype(np.uint8)

    # prev_syn_s = syn_s.copy()
    # syn_s = w * syn_s + (1 - w) * org_s
    # syn_s = syn_s.astype(np.uint8)

    # print(syn_h.dtype)
    # print(syn_s.dtype)
    # print(syn_v.dtype)
    im_syn_hsv = cv2.merge([syn_h, syn_s, syn_v_im.astype(np.uint8)])
    im_syn = cv2.cvtColor(im_syn_hsv, cv2.COLOR_HSV2RGB)

    final_syn_hsv = cv2.merge([syn_h, syn_s, syn_v])
    final_syn = cv2.cvtColor(final_syn_hsv, cv2.COLOR_HSV2RGB)
    print("Type of final_syn: ", type(final_syn))
    print("Data type of final_syn: ", final_syn.dtype)

    final_syn = Image.open(final_syn)

    # S3 upload
    s3 = boto3.client('s3')

    AWS_BUCKET_NAME = 'archdica-conversion'
    AWS_DEFAULT_REGION = env('AWS_DEFAULT_REGION')

    key = os.path.basename(room_image_url)

    s3.upload_fileobj(final_syn, AWS_BUCKET_NAME, key, ExtraArgs={'ACL': 'public-read'})

    object_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format(AWS_BUCKET_NAME, AWS_DEFAULT_REGION, key)

    return object_url
