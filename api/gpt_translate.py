"""
GitHub 教學的是使永 langchain 但這邊就還是直接使用 Azure openAI 來做範例
第一個就是簡單介紹怎麼打 API

順便突然發現我們的 API 可以打圖片
"""

import os
import sys
import yaml  
import logging  
from openai import AzureOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import AZURE_CONFIG, LANGUAGE_LIST, SOURCE_LANGUAGE, SYSTEM_PRMOPT

logger = logging.getLogger(__name__)

class Gpt4oTranslate:
    def __init__(self):
        with open(AZURE_CONFIG, 'r') as file:  
            self.config = yaml.safe_load(file)  

        self.client = AzureOpenAI(api_key=self.config['API_KEY'],
                            api_version=self.config['AZURE_API_VERSION'],
                            azure_endpoint=self.config['AZURE_ENDPOINT'],
                            azure_deployment=self.config['AZURE_DEPLOYMENT']
                            )
        
    def translate(self, sourse_text, sourse_lang, target_lang):
        if {sourse_lang, target_lang}.issubset(LANGUAGE_LIST):  
            sourse_lang_inex = LANGUAGE_LIST.index(sourse_lang)
            target_lang_inex = LANGUAGE_LIST.index(target_lang)

            source_language = SOURCE_LANGUAGE[sourse_lang_inex][target_lang_inex]
            system_prompt = SYSTEM_PRMOPT[target_lang]
            system_prompt = system_prompt.replace("source_language", source_language)  
            logger.debug(f"system prompt: {system_prompt}")
        
        # 調用 OpenAI 模型  
        response = self.client.chat.completions.create(
            model=self.config['AZURE_DEPLOYMENT'], 
            messages=[
                { "role": "system", "content": system_prompt},
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": sourse_text
                    }
                ] } 
            ],
            max_tokens=4000 
        )  
        
        return response.choices[0].message.content