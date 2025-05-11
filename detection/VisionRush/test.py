#!/usr/bin/env python
# -*- coding:utf-8 -*-
import requests
import json
import requests
import json

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
}

url = "http://127.0.0.1:10005/inter_api"
image_path = './dataset/val_dataset/51aa9b8d0da890cd1d0c5029e3d89e3c.jpg'
data_map = {'img_path':image_path}
response = requests.post(url, data=json.dumps(data_map), headers=header)
content = response.content
print(json.loads(content))