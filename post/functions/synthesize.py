import os
import cv2
import numpy as np
from PIL import Image
from .pytorch_room_layout.XiaohuLuVPDetection.lu_vp_detect.vp_detection import VPDetection
import boto3
from io import BytesIO
from django.conf import settings
from urllib.parse import urlparse
from decouple import config
import time

#     Refering for one image data   #
def synthesize(org_image, refer_image, type, scale_factor=None, show_img=False):

    room_image_url = org_image

    try:
        object_ = type

        length_thresh = 70
        principal_point = None
        focal_length = 1300  # 1102.79
        seed = 1300

        vpd = VPDetection(length_thresh, principal_point, focal_length, seed)

        #       Layout Part       #
        # img = Image.open(layout_path + image)
        # # print(type(img))
        # img_np = np.invert(np.asarray(img))
        # # print(img_np.max(), img_np.min())
        # ret, thr = cv2.threshold(img_np, 254, 255, cv2.THRESH_BINARY_INV)

        # S3 client
        client = boto3.client('s3',
                              aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
                              aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
                              region_name=config('AWS_DEFAULT_REGION'))

        # S3 key of image
        key = os.path.basename(org_image)

        filename, file_extension = os.path.splitext(key)
        if file_extension is not '.png':
            file_extension = '.png'

        key = filename + file_extension
        print("key: ", key)

        # Load mask image from S3
        if object_ == "wall":

            bucket = 'wall-mask'

            file_byte_string = client.get_object(Bucket=bucket, Key=key)['Body'].read()
            org_image = Image.open(BytesIO(file_byte_string))

        elif object_ == "floor":

            bucket = 'floor-mask'

            file_byte_string = client.get_object(Bucket=bucket, Key=key)['Body'].read()
            org_image = Image.open(BytesIO(file_byte_string))

        else:
            return

        #       Mask Part         #
        org_color_np, org_np = np.split(np.asarray(org_image), 2, axis=1)

        print("shape of org_color_np: ", org_color_np.shape)
        print("shape of org_np: ", org_np.shape)

        # plt.subplot(131)
        # plt.imshow(org)
        # plt.show()
        img_size = (org_np.shape[1], org_np.shape[0])

        # Load reference image from S3
        bucket = 'archdica-material'

        # s3 config
        parse_result = urlparse(refer_image[1:])
        print("parse_result: ", parse_result)

        key = parse_result.path[1:]
        print("s3 refer key: ", key)

        # load refer image from s3
        try:
            file_byte_string = client.get_object(Bucket=bucket, Key=key)['Body'].read()
            refer_image = Image.open(BytesIO(file_byte_string))
        except Exception as e:
            print("Error: ", e)

        # refer = Image.open(refer_path)
        refer = np.asarray(refer_image)
        print("shape of refer: ", refer.shape)

        if refer.shape[2] > 3:
            refer = refer[:, :, :3]
        print("converted shape of refer: ", refer.shape)

        #     refer_size와 img_size가 동일하거나 refer_size가 작은 경우를 고려해야한다.     #
        print('scale_factor :', scale_factor)
        refer = np.tile(refer, (scale_factor, scale_factor, 1))
        # print(refer.shape)
        # size_ratio = math.floor(min((refer.shape[0] / (org_np.shape[0] * 1.5)), (refer.shape[1] / (org_np.shape[1] * 1.5))))
        size_ratio = min((refer.shape[0] / (org_np.shape[0] * 1.5)), (refer.shape[1] / (org_np.shape[1] * 1.5)))
        print('size_ratio :', size_ratio)
        refer = Image.fromarray(refer).resize((int(refer.shape[1] / size_ratio), int(refer.shape[0] / size_ratio)))

        org_color_np2 = org_color_np.copy()
        org_np = org_np.astype(np.uint8)
        org_np2 = org_np.copy()
        org_np3 = org_np.copy()

        start = time.time()

        #                   Find Best Vline_list                #
        kernel = np.ones((10, 10), np.uint8)

        org_np2_morp = cv2.morphologyEx(org_np2, cv2.MORPH_CLOSE, kernel)
        org_np2_copy = org_np2_morp.copy()
        # org_np2_copy_gray = cv2.cvtColor(org_np2_copy, cv2.COLOR_RGB2GRAY)

        ret, thr_org_np2 = cv2.threshold(org_np2_copy, 127, 255, cv2.THRESH_BINARY)
        thr_org_np2 = cv2.morphologyEx(thr_org_np2, cv2.MORPH_CLOSE, kernel)
        thr_org_np2_copy = thr_org_np2.copy()

        if object_ == 'wall':
            edge_org_np2 = cv2.Canny(org_np2_morp, 20, 100)
            edge_org_np2_copy = edge_org_np2.copy()
            edge_org_np2_copy2 = cv2.Canny((org_np2_copy / 255. * org_np2_copy).astype(np.uint8), 20, 60)

            reg_xs = get_vline_points_inborder(vpd, edge_org_np2_copy2)
            print('#            Used vl_list for vline work          #')
            print('edge_org_np2_copy2')
            print()

            #                                     Find Best Hline                                      #
            #     vline 이 존재하지 않는 경우도 고려해야한다.     #
            if len(reg_xs) == 0:
                reg_xs.append([0, 0])
                # reg_ys.append([0, org_np2.shape[0]])

            #         Divide Session by vline       #
            for reg_index, reg_x in enumerate(reg_xs):

                #     In a Session     #
                #     1.  find vanishing point    #
                #     2.  Find 3 points (vp, top & bottom points)     #
                #     3.  Do warfine and attach to the black plane      #

                print("#      Session Status      #")
                print('reg_index, reg_x :', reg_index, reg_x)

                #     0.  crop by vline     #
                #   Find Max_x, min_x, (Max_y, min_y = org.shape[0], 0)
                #   1.    우편에 한해서 max_x = 우편 vline max_x & min_x = 현재 vline min_x
                #   2.    좌편에 한해서 max_x = 현재 Max_x & min_x = 좌편 vline min_x

                #      vline 별로 양옆으로 작업을 하면 len(vline) = 1의 작업을 반복할 필요가 없어진다.    #
                iter = False
                while True:

                    #                   We need Max, min x & y                #
                    # four_inters = list()
                    find_pair = True
                    # centroid_inters = all_centroid_inters[inters_i]

                    if not iter:

                        #       오른쪽 끝 vline 이면        #
                        if reg_index == len(reg_xs) - 1:
                            print('rightest vline')
                            #   1.    우편에 한해서 max_x = 우편 vline max_x & min_x = 현재 vline min_x
                            max_x = org_np.shape[1]
                            min_x = np.min(reg_x)

                        else:
                            print('middle vline')
                            next_reg_x = reg_xs[reg_index + 1]
                            max_x = np.max(next_reg_x)
                            if np.min(reg_x) < 0:
                                min_x = 0
                            else:
                                min_x = np.min(reg_x)

                    #     i = 0 에 한해서만 왼쪽으로도 refering 진행, 나머지는 오른쪽으로만     #
                    else:
                        #   2.    좌편에 한해서 max_x = 현재 Max_x & min_x = 좌편 vline min_x
                        max_x = np.max(reg_x)
                        min_x = 0

                    print('min_x, max_x :', min_x, max_x)
                    # max_y = org_np.shape[0]
                    # min_y = 0

                    #     Make Session      #
                    session = org_np2[:, int(min_x):int(max_x)]
                    # plt.imshow(session)
                    # plt.show()

                    if (max_x - min_x) > 50:
                        length_thresh = 50
                    else:
                        length_thresh = 20
                        if max_x - min_x < 20:
                            #     i != 0 인 경우 break     #
                            if reg_index == 0 and np.sum(reg_xs) != 0 and not iter:
                                iter = True
                                print('iter :', iter)
                                continue
                            else:
                                break

                    principal_point = None
                    focal_length = 1300  # 1102.79
                    seed = None
                    vpd = VPDetection(length_thresh, principal_point, focal_length, seed)

                    try:
                        vpd.find_vps(session)
                        # vps = vpd.vps_2D
                        vl_img, vl_list = vpd.create_debug_VP_image(show_vl=True)

                        #       Closing Morphing by OpenCV      #
                        kernel = np.ones((10, 10), np.uint8)
                        session = cv2.morphologyEx(session, cv2.MORPH_CLOSE, kernel)

                        #     Before Canny    #
                        session_prob = session / 255.

                        #     Multiply mask map with Edge => Erase Unnecessary vl line    #
                        #   multiply one time   #
                        session2 = session_prob * (session)
                        edge_session2 = cv2.Canny(session2.astype(np.uint8), 20, 60)
                        #       two time      #
                        # session2 = session_prob**2 * (session)
                        # edge_session2 = cv2.Canny(session2.astype(np.uint8), 20, 60)

                        vpd.find_vps(edge_session2)
                        # vps = vpd.vps_2D
                        vl_edge_img2, vl_edge_list2 = vpd.create_debug_VP_image(show_vl=True)

                    except Exception as e:
                        print("Error in vpd Sessions Zone :", e)
                        print()
                        #     i != 0 인 경우 break     #
                        if reg_index == 0 and np.sum(reg_xs) != 0 and not iter:
                            iter = True
                            print('iter :', iter)
                            continue

                        else:
                            break

                    #         What is best vl_list      #
                    vl_list = vl_edge_list2
                    print('#            Used vl_list for hline work          #')
                    print('vl_edge_list2')
                    print()

                    h_lines = list()
                    # print('vps :', vps)

                    top_vl = list()
                    bot_vl = list()

                    left_vl_list = list()
                    right_vl_list = list()

                    left_angle = list()
                    right_angle = list()

                    for vl in vl_list:
                        x0, y0, x1, y1 = vl
                        slope = (y1 - y0) / float(x1 - x0)
                        angle = math.degrees(math.atan(slope))

                        if abs(angle) < 70:
                            if (y0 + y1) / 2 < vl_img.shape[0] / 2:
                                # print(y0, y1)
                                top_vl.append(vl)
                                if angle < 0:
                                    left_vl_list.append(vl)
                                    left_angle.append(abs(angle))
                                else:
                                    right_vl_list.append(vl)
                                    right_angle.append(abs(angle))
                            else:
                                bot_vl.append(vl)
                                if angle > 0:
                                    left_vl_list.append(vl)
                                    left_angle.append(abs(angle))
                                else:
                                    right_vl_list.append(vl)
                                    right_angle.append(abs(angle))
                            # cv2.line(skl_copy, (int(x1), int(y1)), (int(x0), int(y0)), (0, 0, 255), 2,
                            #                  cv2.LINE_AA)
                            h_lines.append(vl)
                        #   regression(vl_img, (x0, x1), (y0, y1), color=(0,255,255),axis=1)

                    print('#        Original vl list        #')
                    print('len(left_vl_list) :', len(left_vl_list))
                    print('len(right_vl_list) :', len(right_vl_list))
                    print()

                    #             소실점 방향 선택            #
                    #       original vl_list 는 remove 가 존재하기 때문에 copy_ version 사용한다.       #
                    copy_left_vl_list = left_vl_list.copy()
                    copy_right_vl_list = right_vl_list.copy()
                    print('len(copy_left_vl_list) :', len(copy_left_vl_list))
                    print('len(copy_right_vl_list) :', len(copy_right_vl_list))

                    if len(copy_left_vl_list) >= len(copy_right_vl_list):
                        direction = 'left'
                        vl_list, copy_vl_list, angle = left_vl_list, copy_left_vl_list, left_angle
                    else:
                        direction = 'right'
                        vl_list, copy_vl_list, angle = right_vl_list, copy_right_vl_list, right_angle

                    print("#            Remove Outliered Angle in Hlines           #")
                    remove_outlier_angle(vl_list, copy_vl_list, angle)
                    # remove_outlier_angle(right_vl_list, copy_right_vl_list, right_angle)
                    print()

                    print('len(left_vl_list) :', len(left_vl_list))
                    print('len(right_vl_list) :', len(right_vl_list))
                    print()

                    # if len(left_vl_list) == 0 and len(right_vl_list) == 0:
                    if len(vl_list) == 0:
                        #     i != 0 인 경우 break     #
                        if reg_index == 0 and not iter:
                            iter = True
                            print('iter :', iter)
                            continue

                        else:
                            break

                    #                       Find External vn_line                     #
                    l2 = Line((0, 0), (0, vl_img.shape[0]))
                    l3 = Line((vl_img.shape[1], 0), (vl_img.shape[1], vl_img.shape[0]))

                    ex_top_vl, ex_bot_vl = get_hline_points_inborder(vl_img, l2, l3, vl_list, top_vl, bot_vl)

                    print()
                    print('#          Extended vl list left / right  TB Condition        #')
                    print('len(ex_top_vl) :', len(ex_top_vl))
                    print('len(ex_bot_vl) :', len(ex_bot_vl))
                    print()

                    #       find min max Line     #
                    toppest_vl, bottest_vl = toppest_bottest_vl(vl_img, ex_top_vl, ex_bot_vl, direction)

                    #       Figure out min_y & max_y of thr_session       #
                    # print('thr_session.shape :', thr_session.shape)
                    gray = cv2.cvtColor(session, cv2.COLOR_RGB2GRAY)
                    ret, thr_session = cv2.threshold(session, 127, 255, cv2.THRESH_BINARY)
                    thr_session = cv2.morphologyEx(thr_session, cv2.MORPH_CLOSE, kernel)
                    thr_session_gray = cv2.cvtColor(thr_session, cv2.COLOR_RGB2GRAY)
                    print('thr_session_gray.shape :', thr_session_gray.shape)

                    min_x_coord, max_x_coord, min_y_coord, max_y_coord, top_parallel, _ = top_bot_mask(thr_session_gray)

                    print('max_x_coord, min_x_coord :', max_x_coord, min_x_coord)
                    print('max_y_coord, min_y_coord :', max_y_coord, min_y_coord)

                    try:

                        #     None 이라면 대칭이동을 통해 만들어주어야한다.     #
                        if len(ex_bot_vl) == 0:
                            if toppest_vl is not None:
                                bottest_vl = line_mirroring(toppest_vl, max_y_coord, l2, l3)
                                print('bottest_vl :', bottest_vl)

                        elif len(ex_top_vl) == 0:
                            if bottest_vl is not None:
                                toppest_vl = line_mirroring(bottest_vl, min_y_coord, l2, l3)
                                print('toppest_vl :', toppest_vl)

                        for (x1, y1, x0, y0) in [toppest_vl, bottest_vl]:
                            cv2.line(vl_img, (int(x1), int(y1)), (int(x0), int(y0)), (0, 0, 255), 3, cv2.LINE_AA)

                    except Exception as e:
                        print('Error in drawing top, bot -est lines :', e)

                        #     i != 0 인 경우 break     #
                        if reg_index == 0 and not iter:
                            iter = True
                            print('iter :', iter)
                            continue

                        else:
                            break

                    #               Choose 4 points             #
                    tl, tr, br, bl = choose_4points(toppest_vl, bottest_vl, min_x_coord, l2, l3, top_parallel)

                    #       Refering      #
                    #   tl, tr, br, bl    #
                    #     refer를 위해 src_x => 0 으로 맞춰준다.    #
                    src = np.array([
                        [0, 0],
                        [vl_img.shape[1], 0],
                        [vl_img.shape[1], vl_img.shape[0]],
                        [0, vl_img.shape[0]]], dtype="float32")
                    dst = np.array([list(tl),
                                    list(tr),
                                    list(br),
                                    list(bl)], dtype="float32")

                    print()
                    print("#          Warping Points        #")
                    print('src :', src)
                    print('dst :', dst)

                    refered = crop_and_warp(refer, vl_img, src, dst)

                    org_color_np2[:, int(min_x):int(max_x)] = refered

                    #     i != 0 인 경우 break     #
                    if reg_index == 0 and np.sum(reg_xs) != 0 and not iter:
                        iter = True
                        print('iter :', iter)

                    else:
                        break

        elif object_ == 'floor':

            # print(thr_org_np2.dtype)
            thr_org_np2 = cv2.cvtColor(thr_org_np2, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(thr_org_np2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            min_height_point = np.array([[thr_org_np2.shape[0], thr_org_np2.shape[1]]])
            # print(min_height_point)
            for contour in contours:
                # print(contour.shape)
                for point in contour:
                    # print(point[0][1], min_height_point[0][1])
                    if 0 < point[0][1] <= min_height_point[0][1]:
                        # print(min_height_point[0][1])
                        min_height_point = point

            dst = np.float32([min_height_point[0], [0, thr_org_np2.shape[1]], [thr_org_np2.shape[0], thr_org_np2.shape[1]]])
            src = np.float32(
                [[min_height_point[0][0], 0], [0, thr_org_np2.shape[1]], [thr_org_np2.shape[0], thr_org_np2.shape[1]]])
            matrix = cv2.getAffineTransform(src, dst)

            refer = np.asarray(refer)
            refer = cv2.warpAffine(refer, matrix, (refer.shape[1], refer.shape[0]))
            row, col, _ = org_color_np2.shape
            org_color_np2 = refer[:row, :col]

        #         Refer의 검은 부분은 original image로 채운다.       #
        org_color_np2 = np.where(org_color_np2 == 0, org_color_np, org_color_np2)

        print('elapsed time :', time.time() - start)
        print()

        # print('np.max(org_np) :', np.max(org_np))
        org_np = org_np / 255.
        refer = org_color_np2
        # plt.show()

        #     1.wall_mask * reference color image + (1 - wall_mask) * original_image)    #
        syn_ = org_np * (refer) + (1 - org_np) * org_color_np
        syn = org_np * syn_ + (1 - org_np) * org_color_np

        #       Brightness Preservation      #
        org_hsv = cv2.cvtColor(np.uint8(org_color_np), cv2.COLOR_RGB2HSV)
        syn_hsv = cv2.cvtColor(np.uint8(syn), cv2.COLOR_RGB2HSV)

        org_h, org_s, org_v = cv2.split(org_hsv)
        syn_h, syn_s, syn_v = cv2.split(syn_hsv)

        mask_map = org_np[:, :, 0]
        # print(mask_map.shape)

        w = org_v / 255.
        w = w / 1.87
        syn_v2 = mask_map * syn_v + (1 - mask_map) * org_v
        syn_v2 = (1 - w) * syn_v2 + w * org_v
        syn_v2 = syn_v2.astype(np.uint8)

        final_syn_hsv = cv2.merge([syn_h, syn_s, syn_v2])
        final_syn = cv2.cvtColor(final_syn_hsv, cv2.COLOR_HSV2RGB)

        if show_img:
            plt.figure(figsize=(15, 10))

            plt.subplot(141)
            plt.imshow(org_np3)
            plt.axis('off')

            plt.subplot(142)
            plt.imshow(org_color_np2)
            plt.axis('off')

            plt.subplot(143)
            plt.imshow(final_syn)
            plt.axis('off')

            plt.subplot(144)
            plt.imshow(org_color_np)
            plt.axis('off')
            plt.show()

        # convert ndarray to file-like object for s3 upload
        img = Image.fromarray(final_syn)
        img_obj = BytesIO()
        img.save(img_obj, format="jpeg")
        img_obj.seek(0)

        # S3 upload
        s3 = boto3.client('s3')

        AWS_BUCKET_NAME = 'archdica-conversion'
        AWS_DEFAULT_REGION = settings.AWS_REGION

        key = os.path.basename(room_image_url)

        s3.upload_fileobj(img_obj, AWS_BUCKET_NAME, key, ExtraArgs={'ACL': 'public-read'})

        object_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format(AWS_BUCKET_NAME, AWS_DEFAULT_REGION, key)
    except Exception as e:
        print("Error: ", e)

    return object_url