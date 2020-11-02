import subprocess


def execute(room_image_url):
    wall_cmd = 'python3 -u ./post/SSP/test.py --imgs {0} --gpu 0 --cfg ./post/SSP/config/ade20k-hrnetv2.yaml TEST.result ./post/SSP/test_result/wall/ TEST.checkpoint epoch_0.pth MODEL.object_index 0'.format(
        room_image_url)
    floor_cmd = 'python3 -u ./post/SSP/test.py --imgs {0} --gpu 0 --cfg ./post/SSP/config/ade20k-hrnetv2.yaml TEST.result ./post/SSP/test_result/floor/ TEST.checkpoint epoch_0.pth MODEL.object_index 3'.format(
        room_image_url)
    subprocess.call(wall_cmd, shell=True)
    subprocess.call(floor_cmd, shell=True)

    return
