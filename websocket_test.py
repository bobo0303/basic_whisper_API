import asyncio  
import websockets  
import json  
import time  
  
async def test_websocket():  
    uri = "ws://localhost:80/ws/rtt_translate"  
      
    async with websockets.connect(uri) as websocket:  
        # 构建要发送的测试请求数据  
        transcription_request_zh = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_1",  
            "times": "2025-01-01 00:00:00",  
            "o_lang": "en",  
            "t_lang": "zh"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_zh))  
        await websocket.send(audio_data)  
          
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
                    
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)  
          
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
                    
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)  
          
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
                    
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)  
          
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)            
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)            
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)            
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
        
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)          
        
        time.sleep(1)
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:10",  
            "o_lang": "en",  
            "t_lang": "ja"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)  
        
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:11",  
            "o_lang": "en",  
            "t_lang": "de"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)  
        
        transcription_request_ja = {  
            "meeting_id": "test_meeting",  
            "device_id": "test_device",  
            "audio_uid": "test_audio_uid_2",  
            "times": "2025-01-01 00:00:02",  
            "o_lang": "en",  
            "t_lang": "ko"  
        }  
          
        # 读取音频文件  
        with open("/mnt/audio/test.wav", "rb") as f:  
            audio_data = f.read()  
          
        # 发送请求数据  
        await websocket.send(json.dumps(transcription_request_ja))  
        await websocket.send(audio_data)  
          
        while True:  
            # 持续接收并打印服务器的响应  
            response = await websocket.recv()  
            print(f"Received response: {response}")  
  
# 运行测试  
asyncio.get_event_loop().run_until_complete(test_websocket())  