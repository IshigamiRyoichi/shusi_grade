from ast import IsNot
from bs4 import BeautifulSoup
from numpy import imag
import requests
import os
import cv2
import tempfile

def get_url(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'lxml')
    image_tag = soup.select('img')
    # print(image_tag)
    return image_tag

def create_image_tag_list(image_tag):
    image_list = []
    for img in image_tag:
        img_str = str(img)
        # print(img_str)
        pos2 = 0
        pos1 = img_str.find('src="') + 5
        if '.png' in img_str:
            pos2 = img_str.find('.png') + 4
        elif '.jpg' in img_str:
            pos2 = img_str.find('.jpg') + 4
        if pos2 != 0:
            image_list.append(img_str[pos1:pos2])
    print(image_list)
    return image_list

def imread_web(url):
    image = None
    # エラー処理
    try:
        res = requests.get(url)
        # tempfileを作成し、読み込む
        fp = tempfile.NamedTemporaryFile(dir='./', delete=False)
        fp.write(res.content)
        fp.close()
        image = cv2.imread(fp.name)
        os.remove(fp.name)
    except requests.exceptions.RequestException as error:
        print("エラー:",error)
    return image

def use_openCV(image_list):
    os.makedirs('./use_openCV',exist_ok=True)
    output_path = './use_openCV/'
    for i,image_data in enumerate(image_list):
        web_image = imread_web(image_data)
        file_name = os.path.join(output_path, 'image'+str(i)+'.jpg')
        # print(web_image)
        if web_image is not None:
            cv2.imwrite(str(file_name), web_image)

        
if __name__ =='__main__':
    url = 'https://www.sanrio.co.jp/'
    image_tag = get_url(url)
    image_list = create_image_tag_list(image_tag)
    use_openCV(image_list)

