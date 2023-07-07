import requests


"""
image를 받아서 app.py에 predict하도록 이미지 경로 넘겨줌
"""

# API endpoint URL
CLASSIFICATION_MODEL_API_URL = 'http://127.0.0.1:8080/predict'

# Image file
img_path = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/DL/Flask_APP/img.png'


with open(img_path, 'rb') as f:
    files = {'image': f}
    requests = requests.post(CLASSIFICATION_MODEL_API_URL,files=files)
    
# Check request
# predict함수에서 요청의 상태 코드가 200인지 확인
if requests.status_code == 200:     
    try:
        prediction = requests.json()['predictions']
        print("Predict result: ",prediction)

    except Exception as e:
        print("API ERROR: ", str(e))

else:
    print("API ERROR!!", requests.text)