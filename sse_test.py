import requests  
import time
import json  

# 定義服務的 URL  
POST_URL = "http://127.0.0.1:80/sse_rtt_translate/v2"  
GET_URL = "http://127.0.0.1:80/sse_rtt_translate"  

# 構建 POST 請求的數據  
files = {  
    'file': ('test_audio.wav', open('audio/test.wav', 'rb'), 'audio/wav')  
}  

transcription_request = {  
    'meeting_id': '123',  
    'device_id': '456',  
    'audio_uid': '789',  
    'times': '2023-10-05T12:34:56',  
    'o_lang': 'en',  
    't_lang': 'de'  
}  

# 發送 POST 請求  
response = requests.post(POST_URL, files=files, data=transcription_request)  
print(f"POST response status: {response.status_code}")  

try:  
    post_response_data = response.json()  
    print(f"POST response data: {post_response_data}")  
except json.JSONDecodeError:  
    print("POST response is not in JSON format") 

transcription_request = {  
    'meeting_id': '123',  
    'device_id': '4526',  
    'audio_uid': '7849',  
    'times': '2023-10-05T12:34:58',  
    'o_lang': 'en',  
    't_lang': 'ja'  
}  

files = {  
    'file': ('test_audio.wav', open('audio/test.wav', 'rb'), 'audio/wav')  
}  

# 發送 POST 請求  
response = requests.post(POST_URL, files=files, data=transcription_request)  
print(f"POST response status: {response.status_code}")  

try:  
    post_response_data = response.json()  
    print(f"POST response data: {post_response_data}")  
except json.JSONDecodeError:  
    print("POST response is not in JSON format")  

# 確認請求已添加到等待列表後，開始接收 SSE 事件  
# time.sleep(1)  # 等待一秒鐘，確保請求已被處理  

# 發送 GET 請求以接收 SSE 事件  
response = requests.get(GET_URL, stream=True)  

for line in response.iter_lines():  
    if line:  
        decoded_line = line.decode('utf-8')  
        print(f"SSE event data: {decoded_line}")  
        
# # 定義服務的 URL，並在查詢參數中包含語言參數  
# POST_URL = "http://127.0.0.1:80/sse_rtt_translate?o_lang=en&t_lang=de&meeting_id=123&device_id=456&audio_uid=789&times=2023-10-05T12:34:56"  
# GET_URL = "http://127.0.0.1:80/sse_rtt_translate"  
  
# # 構建 POST 請求的數據  
# files = {  
#     'file': ('test_audio.wav', open('audio/test.wav', 'rb'), 'audio/wav')  
# }  
  
# # 發送 POST 請求  
# response = requests.post(POST_URL, files=files)  
# print(f"POST response status: {response.status_code}")  

# try:  
#     post_response_data = response.json()  
#     print(f"POST response data: {post_response_data}")  
# except json.JSONDecodeError:  
#     print("POST response is not in JSON format")  
  
# # 確認請求已添加到等待列表後，開始接收 SSE 事件  
# time.sleep(1)  # 等待一秒鐘，確保請求已被處理  
  
# # 發送 GET 請求以接收 SSE 事件  
# response = requests.get(GET_URL, stream=True)  
  
# for line in response.iter_lines():  
#     if line:  
#         decoded_line = line.decode('utf-8')  
#         print(f"SSE event data: {decoded_line}") 