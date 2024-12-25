import requests

# 設定 API 端點 URL
url = "http://0.0.0.0:52001/transcribe"  # 這裡修改成你的 FastAPI 服務的 URL

# 讀取要傳輸的音頻檔案（二進制格式）
with open("audio/test.wav", "rb") as f:
    audio_binary_file = f.read()

    files = {"file": ("test.wav", audio_binary_file)}

    # 發送 POST 請求到 API 端點
    response = requests.post(url, files=files)

    # 解析回應
    if response.status_code == 200:
        command_number = response.text
        print(f"Command number: {command_number}")
    else:
        print(f"Error: {response.text}")

#########################################################################################################
"""
from lib.constant import AZURE_CONFIG, LANGUAGE_QUEUE, SOURCE_LANGUAGE, SYSTEM_PRMOPT
sourse_lang = 'ge'
target_lang = 'ja'

if {sourse_lang, target_lang}.issubset(LANGUAGE_QUEUE):  
    souse_lang_inex = LANGUAGE_QUEUE.index(sourse_lang)
    target_lang_inex = LANGUAGE_QUEUE.index(target_lang)

    source_language = SOURCE_LANGUAGE[souse_lang_inex][target_lang_inex]
    system_prompt = SYSTEM_PRMOPT[target_lang]
    system_prompt = system_prompt.replace("source_language", source_language)  
    print(f"system prompt: {system_prompt}")
"""
#########################################################################################################

