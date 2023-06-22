
import matplotlib.pyplot as plt
import librosa
import librosa.display

import os
import glob
import numpy as np
import IPython
import random
from typing import Any
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Audioprocessing(Dataset):
    def __init__(self, audio_path, tranform=None):
        self.path = glob.glob(os.path.join(audio_path,'*','*.wav'))

        #모드와 증강 정의 
        self.MODES = {
            
            'waveshow' : {
                'org' : self.process_org_waveshow,
                'noise' : self.process_noise_waveshow,
                'stretch' : self.process_stretch_waveshow
            },
            'STFT' : {
                'org' : self.process_org_stft,
                'noise' : self.process_noise_stft,
                'stretch' : self.process_stretch_stft,
            },
            'MelSepctorgram' : {
                'org' : self.process_org_melspec,
                'noise' : self.process_noise_melspec,
                'stretch' : self.process_stretch_melspec
            }
        }  



    def __getitem__(self, index):

        #### 파일을 읽을 수 없는 경우 -> error 예외처리
        try: 
            data, sr = librosa.load(self.path[index], sr=22050)

        except Exception:
            print("읽을 수 없는 오디오 파일입니다.")
            return
            
        #폴더, 파일명 추출
        path = self.path[index]
        folder = path.split('\\')
        folder_name = folder[1]
        file_name = folder[2].replace(".wav", "")
        #print(folder_name, file_name)


        #폴더 생성함수
        self.new_folder(folder_name)
        #0-10초구간 잘라내기 함수
        data_section = self.extraction(data,sr)
        
        #모드 list -> for문 
        Mode = ['waveshow','STFT','MelSepctorgram']
        aug = ['org','noise','stretch']

        for mode in Mode:
            for aug_mode in aug:
                print(mode, aug_mode)
                self.MODES[mode][aug_mode](data_section, folder_name, file_name, aug_mode, mode, sr)
                print('complete!')

        




    def new_folder(self,folder_name) : 
        #제출 데이터 
        submission_dir = "./HW/image_extraction_data"
        final_dir = "./HW/final_data"
        for dir_type in ["MelSepctorgram", "STFT", "waveshow"] : 
            
            #음성 데이터 -> 이미지 저장 하는 폴더 
            os.makedirs(
                f"{submission_dir}/{dir_type}/{folder_name}" , exist_ok=True
            )
            
            #이미지 -> 전처리 완료된 이미지 저장 하는 폴더 
            os.makedirs(
                f"{final_dir}/{dir_type}/{folder_name}" , exist_ok=True
            )
        

    
    #원본 음성에서 MelSepctrogram, STFT, waveshow 이미지 추출 / 구간 : 0초 ~ 10초 사이
    def extraction(self,data,sr):
        
        #0-10초 구간만 구하기
        start_time = 0
        end_time=10
        start_sample = sr * start_time
        end_sample = sr * end_time
        data_section = data[start_sample: end_sample]
        print("구간 자르기 성공")
        return data_section
        
    #origin waveshow
    def process_org_waveshow(self,data_section, folder_name, file_name, aug_mode, mode, sr) :
        print('process_org_waveshow')
        # waveshow 원본 데이터 
        plt.figure(figsize=(12,4))
        librosa.display.waveshow(data_section, color="purple")
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)

    #waveshow + noise
    def process_noise_waveshow(self,data_section, folder_name, file_name, aug_mode, mode, sr) :
        print('process_noise_waveshow')
        # 노이즈 추가 
        noise = 0.05 * np.random.randn(*data_section.shape)
        data_noise = data_section + noise
        
        plt.figure(figsize=(12,4))
        librosa.display.waveshow(data_noise, color="purple")
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)

    #waveshoe + stretch
    def process_stretch_waveshow(self,data_section, folder_name, file_name, aug_mode, mode, sr) :
        
        # stretch 추가 
        data_stretch = librosa.effects.time_stretch(data_section, rate=0.8)
        
        plt.figure(figsize=(12,4))
        librosa.display.waveshow(data_stretch, color="purple")
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close() 

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)

        
    def process_org_stft(self,data_section, folder_name, file_name, aug_mode, mode, sr) :
        
        # stft 계산 
        stft = librosa.stft(data_section)
        
        # stft -> dB 결과 변환 
        stft_db = librosa.amplitude_to_db(abs(stft))
        
        # stft 원본 데이터 
        plt.figure(figsize=(12,4))
        librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)

        
    def process_noise_stft(self,data_section, folder_name, file_name, aug_mode, mode, sr) :
        
        # noise 
        noise_stft = 0.005 * np.random.randn(*data_section.shape)
        noise_stft_data = data_section + noise_stft
        
        # stft 계산 
        stft_noise = librosa.stft(noise_stft_data)
        
        # stft -> dB 결과 변환 
        stft_db_noise = librosa.amplitude_to_db(abs(stft_noise))
        
        # stft 원본 데이터 
        plt.figure(figsize=(12,4))
        librosa.display.specshow(stft_db_noise, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)
        


    def process_stretch_stft(self,data_section, folder_name, file_name, aug_mode, mode, sr) :
        

        # stretching 기법 적용
        rate_stft = 0.8 + np.random.random() * 0.4 # 0.8 ~ 1.2 사이의 랜덤한 비율로 Time stretching
        stretch_data_section = librosa.effects.time_stretch(
            data_section, rate=rate_stft
        )
        
        # stft 계산 
        stft_stretch = librosa.stft(stretch_data_section)
        
        # stft -> dB 결과 변환 
        stft_db_strtch = librosa.amplitude_to_db(abs(stft_stretch))
        
        # stft 원본 데이터 
        plt.figure(figsize=(12,4))
        librosa.display.specshow(stft_db_strtch, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)
    

    def process_org_melspec(self,data_section, folder_name, file_name, aug_mode, mode, sr) :
        
        # stft 계산 
        stft_mel = librosa.stft(data_section)
        
        # 멜 스펙트로그램 계산 
        mel_spec = librosa.feature.melspectrogram(S=abs(stft_mel))
        
        # dB 변환
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        
        plt.figure(figsize=(12,4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)


    def process_noise_melspec(self,data_section, folder_name, file_name, aug_mode, mode, sr) : 
        # stft 계산 
        stft_noise = librosa.stft(data_section)
        
        # 멜 스펙트로그램 계산
        mel_spec_noise = librosa.feature.melspectrogram(S=abs(stft_noise))
        
        # dB 변환
        mel_spect_noise_db = librosa.amplitude_to_db(mel_spec_noise, ref=np.max)
        
        # noise 추가 
        mel_noise = 0.005 * np.random.randn(*mel_spect_noise_db.shape)
        aug_noise_mel = mel_spect_noise_db + mel_noise 
        
        # dB 변환
        aug_noise_db = librosa.amplitude_to_db(aug_noise_mel, ref=np.max)
        
        # 시각화 
        plt.figure(figsize=(12,4))
        librosa.display.specshow(aug_noise_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)

    def process_stretch_melspec(self,data_section, folder_name, file_name, aug_mode, mode, sr) : 
        rate_mel = np.random.uniform(low=0.8, high=1.2)
        stretched_mel = librosa.effects.time_stretch(data_section, rate=rate_mel)
        
        # stft 계산 
        stft_mel_stretch = librosa.stft(stretched_mel)
        
        # 멜 스펙트로그램 계산 
        mel_spec_stretch = librosa.feature.melspectrogram(S=abs(stft_mel_stretch))
            
        # dB 변환
        mel_spec_stretch_db = librosa.amplitude_to_db(mel_spec_stretch, ref=np.max)
        
        # 시각화
        plt.figure(figsize=(12,4))
        librosa.display.specshow(mel_spec_stretch_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png",
                bbox_inches='tight', pad_inches=0)
        plt.close()

        #resize 이미지 저장
        path = f"./HW/image_extraction_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        self.resize(path,mode,folder_name,file_name,aug_mode)


    def expend2square(self,img):
        width, height = img.size
        print(width,height)
        background_color =(0,0,0)

        if width == height:
            return img
        elif width> height:
            result = Image.new(img.mode, (width,width), background_color)
            result.paste(img, (0, (width-height)//2))
            return result
        else:
            result = Image.new(img.mode, (height,height), background_color)
            result.paste(img, (0, (height-width)//2))
            return result
        
        

    def padding(self,image, new_size):
        print('padding!')
        image_padding = self.expend2square(image)   #padding하기 전 expend2square
        image_padding = image_padding.resize((new_size[0], new_size[1]), Image.ANTIALIAS)   #resize
        # plt.imshow(image_padding)
        # plt.show()

        return image_padding


    #이미지로 저장 함수
    #file_save -> padding -> expend2square
    def resize(self,image_path,mode,folder_name,file_name,aug_mode):
        
        img = Image.open(image_path)                #이미지 부르기
        img_sized = self.padding(img,(255,255))     #불러온 이미지 padding시키기
        #새로 저장할 경로
        save_path =f"./HW/final_data/{mode}/{folder_name}/{file_name}_{aug_mode}.png"
        
        img_sized.save(save_path,'png')             #저장
        
        

    def __len__(self):
        return len(self.path)



if __name__ == '__main__':
    #load data
    audio_path = 'C:/Users/iiile/Vscode_jupyter/raw_data/'
    dataset = Audioprocessing(audio_path)
    for item in dataset:
        pass


'''
    이미지 저장확인
    img_path ='C:/Users/iiile/Vscode_jupyter/data/face2.jpg'
    image_sizing = Audioprocessing(img_path)
    image_sizing.file_save(img_path,'test','face_test')
'''






