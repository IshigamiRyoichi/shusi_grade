import cv2
import glob
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox

def find_object(image):
    # バウンディングボックスの座標を読み取る
    bbox, label, conf = cv.detect_common_objects(image)
    #バウンディングボックスの描画
    # output_image = draw_bbox(image, bbox, label, conf)
    # [y1,x1,y2,x2]と出力
    print(bbox[0])
    # cv2.imwrite('./out_put.jpg', output_image)
    return bbox[0]

def use_glob():
    # file_list = glob.glob('./use_openCV/*.jpg')
    file_list = glob.glob('./103.jpg')
    os.makedirs('./output', exist_ok=True)
    for file_image in file_list:
        # 読み込み
        image = cv2.imread(file_image)
        # 物体検知
        bbox = find_object(image)
        # 画像切り取り
        cut_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # 書き込み
        cv2.imwrite('./test.jpg',cut_image)
    # print(file_list)

if __name__ == '__main__':
    use_glob()