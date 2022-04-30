import os
import shutil
import glob
import random

from tqdm import tqdm
def prepare():
    dir = "mrlEyes_2018_01"
    for dirPath, dirNames, fileNames in os.walk(dir):
        for i in tqdm([f for f in fileNames if f.endswith('.png')]):
            if i.split('_')[4] == '0':
                shutil.copy(src=dirPath + '/' + i,
                            dst=r'./dataset/Close Eyes')
            elif i.split('_')[4] == '1':
                shutil.copy(src=dirPath + '/' + i,
                            dst=r'./dataset/Open Eyes')
def prepare2():
    raw_data = 'mrlEyes_2018_01'
    for dirpath, dirname, filename in os.walk(raw_data):
        for file in tqdm([f for f in filename if f.endswith('.png')]):
            if file.split('_')[4] == '0':
                path = './data/train/closed'
                if not os.path.exists(path):
                    os.makedirs(path)
                shutil.copy(src=dirpath + '/' + file, dst=path)
            elif file.split('_')[4] == '1':
                path = './data/train/open'
                if not os.path.exists(path):
                    os.makedirs(path)
                shutil.copy(src=dirpath + '/' + file, dst=path)

def create_test_closed(source, destination, percent):
    path, dirs, files_closed = next(os.walk(source))
    file_count_closed = len(files_closed)
    percentage = file_count_closed * percent
    to_move = random.sample(glob.glob(source + "/*.png"), int(percentage))

    for f in enumerate(to_move):
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.move(f[1], destination)
    print(f'moved {int(percentage)} images to the destination successfully.')


def create_test_open(source, destination, percent):
    path, dirs, files_open = next(os.walk(source))
    file_count_open = len(files_open)
    percentage = file_count_open * percent
    to_move = random.sample(glob.glob(source + "/*.png"), int(percentage))

    for f in enumerate(to_move):
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.move(f[1], destination)
    print(f'moved {int(percentage)} images to the destination successfully.')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #prepare2()
    create_test_closed('./data/train/closed',
                    './data/test/closed',
                    0.2)

    create_test_open('./data/train/open',
                     './data/test/open',
                     0.2)
