from multiprocessing.spawn import import_main_path
import shutil
import random
import glob
import os

os.makedirs('./use_openCV_CP', exist_ok=True)
image_list = glob.glob('./use_openCV/*')
print(image_list)
random.shuffle(image_list)
print(image_list)
for t in range(10):
    shutil.copy(str(image_list[t]), './use_openCV_CP/')