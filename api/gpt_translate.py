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
from lib.constant import AZURE_CONFIG, LANGUAGE_LIST, SOURCE_LANGUAGE, SYSTEM_PRMOPT, USER_PRMOPT_TITLE, SAMPLE_1, SAMPLE_2, SAMPLE_3

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
        
    def translate(self, source_text, source_lang, target_lang):
        if {source_lang, target_lang}.issubset(LANGUAGE_LIST):  
            system_prompt = SYSTEM_PRMOPT[target_lang]
            try:
                system_prompt = system_prompt.replace("source_language", SOURCE_LANGUAGE[LANGUAGE_LIST.index(source_lang)][LANGUAGE_LIST.index(target_lang)])  
                system_prompt = system_prompt.replace("sample_1", SAMPLE_1[source_lang])  
                system_prompt = system_prompt.replace("sample_2", SAMPLE_2[source_lang])  
                system_prompt = system_prompt.replace("sample_3", SAMPLE_3[source_lang])  
            except Exception as e:
                logger.error(f"Error: {e}")
            logger.debug(f"system prompt: {system_prompt}")
        
        # 調用 OpenAI 模型  
        response = self.client.chat.completions.create(
            model=self.config['AZURE_DEPLOYMENT'], 
            messages=[
                { "role": "system", "content": system_prompt},
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": source_text
                    }
                ] } 
            ],
            max_tokens=4000,
            temperature=0,  
        )  
        
        return response.choices[0].message.content