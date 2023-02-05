from __future__ import print_function
import requests
import json
import cv2

# this only works on local setup

addr = f'http://localhost:8080'
test_url = addr + '/quizz_data'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# send http request with image and receive response
response = requests.get(test_url, headers=headers)
# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}