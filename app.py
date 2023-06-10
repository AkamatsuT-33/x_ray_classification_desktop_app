from flask import Flask, render_template, request, url_for, redirect, flash
import os
from PIL import Image, ImageOps
import numpy as np
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
img = ''
img_name = ''
img_path = ''
error = ''

app = Flask(__name__, static_folder="./static/", instance_relative_config=True)

def image_loader(img, loader, device):
    image = img
    image = loader(image)
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image.to(device)

## ホーム画面
@app.route('/', methods=["GET"])
def home():
    if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) +'/static/images/sample.png') == True:
        os.remove(os.path.dirname(os.path.abspath(__file__)) +'/static/images/sample.png')
    else:
        pass
    global img, img_name, img_path, error
    img = ''
    img_name = ''
    img_path = ''
    error = ''
    return render_template('index.html')

## 胸部X線画像分類アプリページ
@app.route('/x_ray_app', methods=["GET"])
def x_ray_get():
    if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) +'/static/images/sample.png') == True:
        os.remove(os.path.dirname(os.path.abspath(__file__)) +'/static/images/sample.png')
    else:
        pass
    global img, img_name, img_path, error
    img = ''
    img_name = ''
    img_path = ''
    error = ''  
    return render_template('x_ray_app.html')

## アップロードされた画像データの処理
@app.route('/x_ray_app', methods=["POST"])
def x_ray_post():
    global img, img_name, img_path, error
    img = ''
    img_name = ''
    img_path = ''
    error = ''
    img = request.files['image']
    img_name = secure_filename(img.filename)
    #img = img.encode()
    img = base64.b64encode(img.read())
    return render_template('ask_upload.html', img=img, name=img_name)

## ファイル確認
@app.route("/x_ray_app/ask_upload", methods=["POST", 'GET'])
def ask_upload():
    global img, img_name
    if img_name == '':
        error = '画像ファイルが選択されていません\nPNGファイル選択してください'
        return render_template('error.html', error = error)
    else:
        pass
    return render_template('ask_upload.html')

## 画像データ保存と完了画面
@app.route("/x_ray_app/complete", methods=["GET", "POST"])
def complete():
    global img, img_name, img_path
    img_path = os.path.dirname(os.path.abspath(__file__)) + '/static/images/sample.png'
    img_01 = base64.b64decode(img)

    with open(img_path, mode='wb') as f:
        f.write(img_01)

    return render_template('complete.html', img=img_01, img_name=img_name)

## アップロードされた画像の確認   
@app.route("/x_ray_app/image_view", methods=["GET"])
def img_view():
    global img, img_name, img_path
    return render_template('image_view.html', img_path=img_path, name = img_name)

## 予測の実行と表示
@app.route("/x_ray_app/recognized_check", methods=["GET"])
def img_recognized():
    global img_name
    model = torch.load(os.path.dirname(os.path.abspath(__file__)) + '/static/models/VGG16_model_cpu_01.pth')
    model.eval()
    loader = torchvision.transforms.Compose([torchvision.transforms.Resize(224) ,torchvision.transforms.RandomInvert(1), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5),(0.5))])
    m = nn.Softmax(dim=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_x = Image.open(os.path.dirname(os.path.abspath(__file__)) +'/static/images/sample.png').convert('RGB')
    image = image_loader(img_x, loader, device)
    #result = m(model(image))
    ans = np.argmax(m(model(image)).to('cpu').detach().numpy())

    if ans == 0:
        res = 'コロナウイルス肺炎'
    elif ans == 1:
        res = '肺白濁'
    elif ans == 2:
        res = '健康な肺'
    else:
        res = 'ウイルス性肺炎'
         
    return render_template('recognized.html', ans=res, name = img_name, aa=ans, img=img_x)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)