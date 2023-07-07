
import torch
import torchvision.transforms as transforms

from PIL import Image
from flask import Flask, jsonify, request

#.py 파일에서 class,함수 불러오기
from imagenet1000_lables_temp import lable_dicr
from vgg11 import VGG11

"""
외부 모델 vgg11-bbd30...에서 미리 학습된 가중치를 가져온다.
vgg11.py에서 내장 모델vgg11을 가지고 왔지만 pretrained=False로 가중치는 가지고 오지 않았다.
외부 모델에서 미리 학습된 가중치를 사용하여 모델을 초기화하고 새로운 작업에 맞게 fine-tuning할 수 있다

"""


app = Flask(__name__)



# Load model
def load_model(model_path):
    model = VGG11(num_classes=1000)
    # 미리 학습된 가중치 로드 -> 모델 초기화, fine-tunning
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

model_path = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/DL/Flask_APP/vgg11-bbd30ac9.pth'
model = load_model(model_path)
#print(model)


# image preprocessing
def preprocessing_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    image = transform(image).unsqueeze(0)


# Set API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    img = Image.open(image)
    img = preprocessing_image(img)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predcited = torch.max(outputs.data, 1)

        # 예측한 것 dict찾고 출력하기
        label_num = int(predcited.item())
        class_name = lable_dicr[label_num]

        prediction = str(class_name)

    # 200응답냄
    return jsonify({'predictions': prediction}),200     


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)