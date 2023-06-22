
import matplotlib.pyplot as plt
import librosa
import librosa.display

import os
import glob
import numpy as np
import IPython
import random

from PIL import Image

'''# 이미지 비율에 맞추기
def expend2square(pil_img, backgoround_color):
    width, height = pil_img.size

    if width == height:
        return pil_img
    
    elif width > height:
        result = Image.new(pil_img, (width,width), backgoround_color)
        result.paste(pil_img,(0, width-height)//2)
        return result
    else:
        result = Image.new(pil_img, (height,height), backgoround_color)
        result.paste(pil_img,(0, height-width)//2)
        return result

#padding 넣어주기
def resize_with_padding(pil_img, new_size, backgoround_color):
    img= expend2square(pil_img, backgoround_color)
    img = img.resize((new_size[0], new_size[1], Image.ANTIALIAS))

    return img
'''

audio_path = 'C:/Users/iiile/Vscode_jupyter/raw_data/'

class AudioProcessing(Dataset):
    def __init__(self, audio_path, tranform=None):
        self.path = glob.glob(os.path.join(audio_path,'*',"*",".wav"))
        

    
    def __getitem__(self, index):
        
        data, sr = librosa.load(self.path, sr=22050)
        folder_name, file_name = self.path[index]
        return folder_name, file_name
    
    def __len__(self):
        return len(self.path)
    

if __name__ == '__main__':
    dataset = AudioProcessing(audio_path)

    for item in dataset:
        print(item)
