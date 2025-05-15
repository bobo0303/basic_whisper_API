
# GEMMA 4B (https://huggingface.co/google/gemma-3-4b-it)

import os
import sys
import logging  
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import LANGUAGE_LIST, SOURCE_LANGUAGE, SYSTEM_PRMOPT, USER_PRMOPT_TITLE, SAMPLE_1, SAMPLE_2, SAMPLE_3, ModlePath

logger = logging.getLogger(__name__)

class Gemma4BTranslate:
    def __init__(self):
        self.models_path = ModlePath()
        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.models_path.gemma, device_map="auto").eval()  
        self.processor = AutoProcessor.from_pretrained(self.models_path.gemma) 
        
    def get_translated_text(self, target_lang, text):  
        if target_lang in USER_PRMOPT_TITLE:  
            prompt_title = USER_PRMOPT_TITLE[target_lang]  
            if text.startswith(prompt_title):  
                text = text[len(prompt_title):]  
            return text  
        else:  
            return text
        
    def translate(self, sourse_text, sourse_lang, target_lang):
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
        
            messages=[
                { 
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}] 
                },
                { 
                    "role": "user", 
                    "content": [{ "type": "text", "text": sourse_text}
                        ] 
                    } 
                ]

            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]

            decoded = self.processor.decode(generation, skip_special_tokens=True)  
            decoded = self.get_translated_text(target_lang, decoded)  
            
            return decoded
        else:
            logger.error(f" | Error: sourse_lang \"{sourse_lang}\" or target_lang \"{target_lang}\" not in LANGUAGE_LIST \"{LANGUAGE_LIST}\" | ")
            return sourse_text
        
if __name__ == "__main__":  
    logging.basicConfig(level=logging.DEBUG)  
      
    translator = Gemma4BTranslate()  
  
    source_text = "Hello, how are you?"  
    source_lang = "en"  
    target_lang = "zh"  
  
    translated_text = translator.translate(source_text, source_lang, target_lang)  
    logger.info(f" | Translated Text: {translated_text} | ")  