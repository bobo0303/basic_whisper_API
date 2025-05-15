import os
import sys
import logging  
import yaml
from ollama import Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import LANGUAGE_LIST, SOURCE_LANGUAGE, SYSTEM_PRMOPT, USER_PRMOPT_TITLE, SAMPLE_1, SAMPLE_2, SAMPLE_3, ModlePath

logger = logging.getLogger(__name__)
 
class OllamaChat:
    def __init__(self, config_path):
        """初始化 Ollama 聊天客戶端
 
        Args:
            config_path: 配置文件路徑
        """
        self.config_path = config_path
        self.config = self._load_config()
        try:
            self.client = Client(host=self.config["HOST"])
            
            self.client.chat(
                model=self.config["MODEL"],
                messages="",
                format="",
                stream=False,
                keep_alive=-1
            )
        except Exception as e:
            logger.error(f" | initial ollama error: {e} (Maybe ollama serve not started) | ")
            raise e
 
    def _load_config(self):
        """載入配置文件"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_translated_text(self, target_lang, text):  
        if target_lang in USER_PRMOPT_TITLE:  
            prompt_title = USER_PRMOPT_TITLE[target_lang]  
            if text.startswith(prompt_title):  
                text = text[len(prompt_title):]  
            return text  
        else:  
            return text

    def chat(
        self,
        temperature = 0.0,
        stream = False,
        format = "",
        sourse_text = "", 
        sourse_lang = "zh", 
        target_lang = "en"
    ):
        """發送聊天請求並獲取回應
 
        Args:
            prompt: 用戶提問內容
            system_prompt: 系統提示詞
            temperature: 溫度參數，控制隨機性
            stream: 是否使用流式輸出
            format: 輸出格式
 
        Returns:
            如果 stream=True，返回流式響應生成器
            如果 stream=False，返回完整響應
        """
        if {sourse_lang, target_lang}.issubset(LANGUAGE_LIST):  
            system_prompt = SYSTEM_PRMOPT[target_lang]
            try:
                system_prompt = system_prompt.replace("source_language", SOURCE_LANGUAGE[LANGUAGE_LIST.index(sourse_lang)][LANGUAGE_LIST.index(target_lang)])  
                system_prompt = system_prompt.replace("sample_1", SAMPLE_1[sourse_lang])  
                system_prompt = system_prompt.replace("sample_2", SAMPLE_2[sourse_lang])  
                system_prompt = system_prompt.replace("sample_3", SAMPLE_3[sourse_lang])  
            except Exception as e:
                logger.error(f"Error: {e}")
            logger.debug(f" | system prompt: {system_prompt} | ")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sourse_text},
            ]
            try:
                response = self.client.chat(
                    model=self.config["MODEL"],
                    messages=messages,
                    format=format,
                    options={"temperature": temperature},
                    stream=stream,
                    keep_alive=-1
                )

                decoded = self.get_translated_text(target_lang, response.message.content)  
            except Exception as e:
                logger.error(f" | ollama Error: {e} | ")
                decoded = sourse_text
                
            return decoded
        else:
            logger.error(f" | Error: sourse_lang \"{sourse_lang}\" or target_lang \"{target_lang}\" not in LANGUAGE_LIST \"{LANGUAGE_LIST}\" | ")
            return sourse_text
        
    def close(self):
        self.client.chat(
            model=self.config["MODEL"],
            messages="",
            format="",
            options={"temperature": 0.0},
            stream=False,
            keep_alive=0
        )        
        logger.info("OllamaChat client closed.")
        
if __name__ == "__main__":  
    import time
    # 配置文件的路徑  
    config_path = "/mnt/lib/gemma_12b_qat.yaml"  
  
    start = time.time()
    # 初始化 OllamaChat 客戶端  
    chat_client = OllamaChat(config_path)  
    end = time.time()
    print("initialization time:", end - start)
  
    # 測試數據  
    sourse_text = "这是一个测试句子。"  
    sourse_lang = "zh"  
    target_lang = "en"  
    
    start = time.time()
    # 呼叫 chat 方法並獲取回應  
    response = chat_client.chat(  
        temperature=0.0,  
        stream=False,  
        format="",  
        sourse_text=sourse_text,  
        sourse_lang=sourse_lang,  
        target_lang=target_lang  
    )  
    end = time.time()
    print("translation time:", end - start)
  
    # 打印回應  
    print("Translated Text:", response)  
    
    start = time.time()
    # 呼叫 chat 方法並獲取回應  
    response = chat_client.chat(  
        temperature=0.0,  
        stream=False,  
        format="",  
        sourse_text=sourse_text,  
        sourse_lang=sourse_lang,  
        target_lang=target_lang  
    )  
    end = time.time()
    print("translation time:", end - start)
  
    # 打印回應  
    print("Translated Text:", response)  
    
    start = time.time()
    # 呼叫 chat 方法並獲取回應  
    response = chat_client.chat(  
        temperature=0.0,  
        stream=False,  
        format="",  
        sourse_text=sourse_text,  
        sourse_lang=sourse_lang,  
        target_lang=target_lang  
    )  
    end = time.time()
    print("translation time:", end - start)
  
    # 打印回應  
    print("Translated Text:", response)  
    
    chat_client.close()