import os
import torch
import timm
import requests
from requests.adapters import HTTPAdapter


# 设置环境变量以禁用 SSL 验证
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# 创建一个会话并禁用 SSL 验证
session = requests.Session()
session.verify = False

# 将会话应用到 huggingface_hub
from huggingface_hub import hf_hub_download

def load_model_with_disabled_ssl():
    try:
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        return model
    except Exception as e:
        print("加载模型时出错：", e)

model = load_model_with_disabled_ssl()
