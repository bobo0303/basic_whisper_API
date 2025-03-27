import os  
import re
import gc  
import time  
import json
import torch
import logging  
from pydub import AudioSegment  

import argostranslate.translate  
# from api.qwen_translate import QWEN7BTranslate  
from api.gemma_translate import Gemma4BTranslate  
from funasr import AutoModel  
from vosk import Model, KaldiRecognizer
from .text_postprocess import extract_sensevoice_result_text

from lib.constant import ModlePath, OPTIONS, IS_PUNC
  
os.environ["ARGOS_DEVICE_TYPE"] = "cuda"  # Set ARGOS to use CUDA  
  
logger = logging.getLogger(__name__)  
  
class Models:  
    def __init__(self):  
        """  
        Initialize the Model class with default attributes.  
        """  
        self.model = None  
        self.model_version = None  
        self.models_path = ModlePath()  
        # self.qwen_translator = QWEN7BTranslate()  
        self.gemma_translator = Gemma4BTranslate()
        self.translate_method = "gemma"  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_parameter = {"model": None,
                                "disable_update": True,
                                "disable_pbar": True,
                                "device": self.device,            
                                }
        
    def load_model(self, model_name):  
        """  
        Load the specified model based on the model's name.  
  
        :param model_name: str  
            The name of the model to be loaded.  
        :rtype: None  
        :logs: Loading status and time.  
        """  
        start = time.time()  
        try:  
            # Release old model resources  
            self._release_model() 
            self.model_version = model_name
            
            if self.model_version == "sensevoice":
                self.model_parameter['model'] = self.models_path.sensevoice
                self.model = AutoModel(**self.model_parameter)
            elif self.model_version.startswith("vosk"):
                self.model = {}
                for field_name, field_value in self.models_path.model_dump().items():  
                    if field_name.startswith("vosk"):                   
                        self.model[field_name] = (KaldiRecognizer(Model(field_value), 16000))
            end = time.time()
                    
            print(f"Model '{self.model_version}' loaded in {end - start:.2f} secomds.")

            if self.model_version == "sensevoice" and IS_PUNC:
                start = time.time()
                print("Start to loading punch model.")
                self.model_parameter['model'] = self.models_path.punc
                self.punc_model = AutoModel(**self.model_parameter)
                end = time.time()
                print(f"Model \'ct-punc\' loaded in {end - start:.2f} secomds.")

            end = time.time()  
            logger.info(f"Model '{self.model_version}' loaded in {end - start:.2f} seconds.")  
        except Exception as e:
            logger.error(f'load_model() models_name:{self.model_version} error:{e}')
            self.model_version = None
  
    def _release_model(self):  
        """  
        Release the resources occupied by the current model.  
  
        :param None: The function does not take any parameters.  
        :rtype: None  
        :logs: Model release status.  
        """  
        if self.model is not None:  
            del self.model  
            gc.collect()  
            self.model = None
            torch.cuda.empty_cache()  
            logger.info("Previous model resources have been released.")  
  
    def change_translate_method(self, method_name):  
        """  
        Change the translation method used by the model.  
  
        :param method_name: str  
            The name of the translation method to be used.  
        :rtype: None  
        """  
        self.translate_method = method_name  
        
    def _convert_to_wav(self, file_path, target_sample_rate=16000):  
        audio = AudioSegment.from_file(file_path)  
        if audio.channels > 1:  
            audio = audio.set_channels(1)  
        audio = audio.set_frame_rate(target_sample_rate)  
        audio.export(file_path, format="wav")  
        
    def _process_transcription(self, transcription):  
        cjk_char_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')  
        result = []  
        i = 0  
        length = len(transcription)  
        
        while i < length:  
            char = transcription[i]  
            
            if char == ' ':  
                if i > 0 and i < length - 1:  
                    prev_char = transcription[i - 1]  
                    next_char = transcription[i + 1]  
                    
                    if cjk_char_pattern.match(prev_char) and cjk_char_pattern.match(next_char):  
                        i += 1  
                        continue  
                    elif cjk_char_pattern.match(prev_char) and next_char.isalpha():  
                        result.append(' ')  
                    elif prev_char.isalpha() and cjk_char_pattern.match(next_char):  
                        result.append(' ')  
                    else:  
                        result.append(char)  
                else:  
                    result.append(char)  
            else:  
                result.append(char)  
            
            i += 1  
        
        return ''.join(result)  
    
    def transcribe(self, audio_file_path, ori):  
        """  
        Perform transcription and translation on the given audio file.  
  
        :param audio_file_path: str  
            The path to the audio file to be transcribed.  
        :param ori: str  
            The original language of the audio.  
        :param tar: str  
            The target language for translation.  
        :rtype: tuple  
            A tuple containing the original transcription, translated transcription, inference time, translation time, and the translation method used.  
        :logs: Inference status and time.  
        """  
        OPTIONS["language"] = ori  
  
        start = time.time()  
        if self.model_version.startswith("vosk"):
            for language in self.model:
                if ori == language[-2:]:    
                    rec = self.model[language]
            self._convert_to_wav(audio_file_path)
            with open(audio_file_path, "rb") as wf:
                wf.read(44) # skip header
                ori_pred = ""
                while True:
                    data = wf.read(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        ori_pred += res["text"]
                res = json.loads(rec.FinalResult())
                ori_pred += res["text"]
            ori_pred = self._process_transcription(ori_pred)  
        else:
            result = self.model.generate(audio_file_path, **OPTIONS)
            ori_pred = result[0]['text']
            if IS_PUNC:
                ori_pred = self.punc_model.generate(input=ori_pred)
                ori_pred = ori_pred[0]['text']
        logger.debug(ori_pred)  
        end = time.time()
        inference_time = end-start

        if self.model_version == 'sensevoice':
            ori_pred = extract_sensevoice_result_text(ori_pred.lower())
            
        logger.debug(f"Inference time {inference_time} seconds.")  
                                                                                                                                            
        return ori_pred, inference_time
  
    def translate(self, ori_pred, ori, tar):
        start = time.time()  
        try:
            if ori != tar and ori_pred != '':
                if self.translate_method == "argos":  
                    ori = 'zt' if ori == 'zh' else ori  
                    tar = 'zt' if tar == 'zh' else tar  
                    translated_pred = argostranslate.translate.translate(ori_pred, ori, tar)
                elif self.translate_method == "gemma":  
                    try:
                        translated_pred = self.gemma_translator.translate(ori_pred, ori, tar)   
                    except Exception as e:
                        translated_pred = ori_pred
                        ori = 'zt' if ori == 'zh' else ori  
                        tar = 'zt' if tar == 'zh' else tar  
                        translated_pred = argostranslate.translate.translate(ori_pred, ori, tar)  
                        logger.error(f'translate() QWEN-7B error:{e}') 
                # elif self.translate_method == "qwen":  
                #     try:
                #         translated_pred = self.qwen_translator.translate(ori_pred, ori, tar)  
                #     except Exception as e:
                #         translated_pred = ori_pred
                #         ori = 'zt' if ori == 'zh' else ori  
                #         tar = 'zt' if tar == 'zh' else tar  
                #         translated_pred = argostranslate.translate.translate(ori_pred, ori, tar)  
                #         logger.error(f'translate() QWEN-7B error:{e}')
            else:  
                translated_pred = ori_pred
        except Exception as e:
            logger.error(f'translate() error:{e}')
            translated_pred = ori_pred
        end = time.time()  
        translate_time = end - start  
        
        return translated_pred, translate_time, self.translate_method
        
        
