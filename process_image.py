import cv2
from scipy import ndimage

# サイズ変更
def resize_image(image_path):
    image = cv2.imread(image_path)
    # 64 * 64に変更
    image_resize = cv2.resize(image, (64,64))
    file_name = './test/image_resize.jpg'
    cv2.imwrite(file_name, image_resize)

# 画像回転
def rotate_image(image_path):
    image = cv2.imread(image_path)
    for ang in [-5, 0, 5]:
        image_rot = ndimage.rotate(image, ang)
        file_name = './test/image_rot' + str(ang) + '.jpg'
        cv2.imwrite(file_name, image_rot)

# 閾値
def threshold_image(image_path):
    image = cv2.imread(image_path)
    # 閾値100を超えたがぞを255にする
    image_thr =  cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO)[1]
    file_name = './test/image_thr.jpg'
    cv2.imwrite(file_name, image_thr)

# ぼかし
def gaussian_image(image_path):
    image = cv2.imread(image_path)
    # 奇数じゃないといけない
    image_gau =  cv2.GaussianBlur(image, (15, 15), 0)
    file_name = './test/image_gau.jpg'
    cv2.imwrite(file_name, image_gau)

# モザイク
def mosaic_image(image_path):
    image = cv2.imread(image_path)
    image_small = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
    image_masaic = cv2.resize(image_small, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    file_name = './test/image_mosaic.jpg'
    cv2.imwrite(file_name, image_masaic)

if __name__ == '__main__':
    image_path = './test1.jpg'
    resize_image(image_path)
    rotate_image(image_path)
    threshold_image(image_path)
    gaussian_image(image_path)
    mosaic_image(image_path)