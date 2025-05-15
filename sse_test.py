import requests  
import sseclient  
import time  
  
# 發送 POST 請求  
audio_file_path = '/mnt/audio/test.wav'  
transcription_request = {  
    "meeting_id": "test_meeting",  
    "device_id": "test_device",  
    "audio_uid": "unique_audio_id",  
    "times": str(int(time.time())),  
    "o_lang": "en",  
    "t_lang": "zh"  
}  
  
files = {  
    'file': open(audio_file_path, 'rb')  
}  
  
response = requests.post("http://127.0.0.1:80/sse_rtt_translate", files=files, data=transcription_request)  
print(response.json())  
  
# 建立 SSE 連接  
response = requests.get("http://127.0.0.1:80/sse_rtt_translate", stream=True)  
client = sseclient.SSEClient(response)  
  
for event in client.events():  
    print(event.data)  