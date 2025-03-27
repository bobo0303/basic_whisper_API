
import os
import sys
import logging  
from transformers import AutoTokenizer, AutoModelForCausalLM  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import LANGUAGE_LIST, SOURCE_LANGUAGE, SYSTEM_PRMOPT, USER_PRMOPT_TITLE, SAMPLE_1, SAMPLE_2, SAMPLE_3, ModlePath

logger = logging.getLogger(__name__)

class QWEN7BTranslate:
    def __init__(self):
        self.models_path = ModlePath()
        self.tokenizer = AutoTokenizer.from_pretrained(self.models_path.qwen)  
        self.model = AutoModelForCausalLM.from_pretrained(self.models_path.qwen).to('cuda')  
        
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
            logger.debug(f"system prompt: {system_prompt}")
        
            messages=[
                    { "role": "system", "content": system_prompt},
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": USER_PRMOPT_TITLE[target_lang]+sourse_text
                        }
                    ] } 
                ]

            # Apply chat template  
            input_ids = self.tokenizer.apply_chat_template(  
                messages,  
                add_generation_prompt=True,  
                return_tensors="pt"  
            ).to('cuda') 

            outputs = self.model.generate(input_ids, max_length=100)  
            response = outputs[0][input_ids.shape[-1]:]  
            translated_text = self.tokenizer.decode(response, skip_special_tokens=True)  
            
            return translated_text
        else:
            logger.error(f"Error: sourse_lang \"{sourse_lang}\" or target_lang \"{target_lang}\" not in LANGUAGE_LIST \"{LANGUAGE_LIST}\"")
            return sourse_text
