
nz = 100        # 잠재공간 벡터 크기
ngf = 64        # 생성자를 통과하는 피쳐데이터 채널 크기
ndf = 64        # 구분자를 통과하는 피쳐데이터 채널 크기
nc = 3          # 이미지 채널 수, RGB


data_root ='GAN/data/cat_dataset'

num_workers = 2
batch_size = 128
img_size = 64
num_epochs = 200
lr = 0.00025

# Adam ->  
beta1 = 0.4