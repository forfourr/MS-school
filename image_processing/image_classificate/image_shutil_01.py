import os
import glob
import shutil
from sklearn.model_selection import train_test_split

#data_path ="C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data"
data_path = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'

class ImageMove:
    def __init__(self, org_folder):
        self.org_folder = org_folder

    def move_image(self):
        file_path_list = glob.glob(os.path.join(self.org_folder, 'final_data',"*","*","*.png"))
        for file_path in file_path_list:
            folder_name = file_path.split('\\')[2]
            if folder_name == 'MelSepctrogram':
                shutil.move(file_path, f"{self.org_folder}/ex_dataset/MelSpectogram")
            elif folder_name == 'STFT':
                shutil.move(file_path, f"{self.org_folder}/ex_dataset/STFT")
            elif folder_name == 'waveshow':
                shutil.move(file_path, f"{self.org_folder}/ex_dataset/waveshow")

# test = ImageMove(data_path)
# test.move_image()


file_path_01 = glob.glob(os.path.join(data_path,"ex_dataset","waveshow","*.png"))
file_path_02 = glob.glob(os.path.join(data_path,"ex_dataset","STFT","*.png"))
file_path_03 = glob.glob(os.path.join(data_path,"ex_dataset","MelSpectogram","*.png"))


# Mel:2997, STFT:2997, waveshow:2997
wave_train_data, wave_val_data = train_test_split(file_path_01, test_size=0.2)
stft_train_data, stft_val_data = train_test_split(file_path_02, test_size=0.2)
mel_train_data, mel_val_data = train_test_split(file_path_03, test_size=0.2)


for wave_train_path in wave_train_data:
    os.makedirs(f"{data_path}/sound/train/waveshow", exist_ok=True)
    shutil.move(wave_train_path, f"{data_path}/sound/train/waveshow")
for wave_val_path in wave_val_data:
    os.makedirs(f"{data_path}/sound/val/waveshow", exist_ok=True)
    shutil.move(wave_val_path,f"{data_path}/sound/val/waveshow")

for stft_train_path in stft_train_data:
    os.makedirs(f"{data_path}/sound/train/STFT", exist_ok=True)
    shutil.move(stft_train_path, f"{data_path}/sound/train/STFT")
for stft_val_path in stft_val_data:
    os.makedirs(f"{data_path}/sound/val/STFT", exist_ok=True)
    shutil.move(stft_val_path,f"{data_path}/sound/val/STFT")


for mel_train_path in mel_train_data:
    os.makedirs(f"{data_path}/sound/train/Mel", exist_ok=True)
    shutil.move(mel_train_path, f"{data_path}/sound/train/Mel")
for mel_val_path in mel_val_data:
    os.makedirs(f"{data_path}/sound/val/Mel", exist_ok=True)
    shutil.move(mel_val_path,f"{data_path}/sound/val/Mel")

print("complete")