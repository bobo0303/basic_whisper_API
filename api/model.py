import os  
import gc  
import time  
import torch
import whisper  
import logging  

from googletrans import Translator  
from funasr import AutoModel  
from queue import Queue  

# from api.gemma_translate import Gemma4BTranslate  
from api.ollama_translate import OllamaChat
from api.gpt_translate import Gpt4oTranslate  

from api.text_postprocess import extract_sensevoice_result_text
from lib.constant import ModlePath, OPTIONS, SV_OPTIONS, SENSEVOCIE_PARMATER, IS_PUNC, PUNC_PARMATER, OLLAMA_MODEL
  
  
logger = logging.getLogger(__name__)  
  
# 配置日誌記錄器設置（如果尚未配置）  
if not logger.handlers:  
    log_format = "%(asctime)s - %(message)s"  
    log_file = "logs/app.log"  
    logging.basicConfig(level=logging.INFO, format=log_format)  
  
    # 創建文件處理器  
    file_handler = logging.handlers.RotatingFileHandler(  
        log_file, maxBytes=10*1024*1024, backupCount=5  
    )  
    file_handler.setFormatter(logging.Formatter(log_format))  
  
    # 創建控制台處理器  
    console_handler = logging.StreamHandler()  
    console_handler.setFormatter(logging.Formatter(log_format))  
  
    logger.addHandler(file_handler)  
    logger.addHandler(console_handler)  
  
logger.setLevel(logging.INFO)  
logger.propagate = False  

class Model:  
    def __init__(self):  
        """Initialize the Model class with default attributes."""  
        self.models_path = ModlePath()  
        # self.gemma_translator = Gemma4BTranslate()  
        self.ollama_translator = OllamaChat(OLLAMA_MODEL['gemma'])  
        self.gpt4o_translator = Gpt4oTranslate()  
        self.google_translator = Translator()  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.model = None  
        self.model_version = None  
        self.punc_model = None  
        self.translate_method = "google"  
        self.processing = None  
        self.result_queue = Queue()  
  
    def load_model(self, models_name):  
        """Load the specified model based on the model's name."""  
        start = time.time()  
        try:  
            # Release old model resources  
            self._release_model()  
            self.model_version = models_name  
  
            # Choose model weight  
            if models_name == "large_v2":  
                self.model = whisper.load_model(self.models_path.large_v2)  
                self.model.to(self.device)  
            elif models_name == "medium":  
                self.model = whisper.load_model(self.models_path.medium)  
                self.model.to(self.device)  
            elif models_name == "sensevoice":  
                self.model = AutoModel(**SENSEVOCIE_PARMATER)  
                if IS_PUNC:  
                    start = time.time()  
                    logger.info(" | Start to loading punch model. | ")  
                    self.punc_model = AutoModel(**PUNC_PARMATER)  
                    end = time.time()  
                    logger.info(f" | Model 'ct-punc' loaded in {end - start:.2f} seconds. | ")  
            end = time.time()  
            logger.info(f" | Model '{models_name}' loaded in {end - start:.2f} seconds. | ")  
        except Exception as e:  
            self.model_version = None  
            logger.error(f' | load_model() models_name: {models_name} error: {e} | ')  
  
    def _release_model(self):  
        """Release the resources occupied by the current model."""  
        if self.model is not None:  
            del self.model  
            gc.collect()  
            self.model = None  
            torch.cuda.empty_cache()  
            logger.info(" | Previous model resources have been released. | ")  
  
    def change_translate_method(self, method_name):  
        """  
        Change the translation method used by the model.  
  
        :param method_name: str  
            The name of the translation method to be used.  
        :rtype: None  
        """  
        if not self.translate_method == method_name and method_name in OLLAMA_MODEL:
            try:
                self.ollama_translator.close()
                self.ollama_translator = OllamaChat(OLLAMA_MODEL[method_name])
                logger.info(f" | old ollama has been released. Initial '{method_name}' has been successful | ")          
            except Exception as e:
                logger.error(f" | ollama translate method change error: {e} | ")
                self.ollama_translator = OllamaChat(OLLAMA_MODEL['gemma'])
                logger.info(f" | Initial the default ollama model 'gemma' | ")          
        self.translate_method = method_name  

    def transcribe(self, audio_file_path, ori):  
        """  
        Perform transcription on the given audio file.  
    
        :param audio_file_path: str  
            The path to the audio file to be transcribed.  
        :param ori: str  
            The original language of the audio.  
        :rtype: tuple  
            A tuple containing the original transcription and inference time.  
        :logs: Inference status and time.  
        """  
        # Set the language option for transcription  
        OPTIONS["language"] = ori  
        SV_OPTIONS["language"] = ori
        
        start = time.time()  # Start timing the transcription process  
    
        if self.model_version == "sensevoice":  
            # Perform transcription using the SenseVoice model  
            result = self.model.generate(audio_file_path, **SV_OPTIONS)  
            ori_pred = result[0]['text']  
            
            if IS_PUNC:  
                # Add punctuation to the transcription if IS_PUNC is enabled  
                ori_pred = self.punc_model.generate(input=ori_pred)  
                ori_pred = ori_pred[0]['text']  
            
            ori_pred = extract_sensevoice_result_text(ori_pred.lower())  # Extract and clean the transcription text  
        else:  
            # Perform transcription using a different model  
            result = self.model.transcribe(audio_file_path, **OPTIONS)  
            logger.debug(result)  # Log the transcription result  
            ori_pred = result['text']  
    
        end = time.time()  # End timing the transcription process  
        inference_time = end - start  # Calculate the time taken for transcription  
    
        logger.debug(f" | Inference time {inference_time} seconds. | ")  # Log the inference time  
    
        return ori_pred, inference_time  # Return the transcription and inference time  
  
    def translate(self, ori_pred, ori, tar):  
        """  
        Translate the given text from the original language to the target language.  
    
        :param ori_pred: str  
            The original text to be translated.  
        :param ori: str  
            The original language of the text.  
        :param tar: str  
            The target language for translation.  
        :return: tuple  
            A tuple containing the translated text, the translation time, and the translation method used.  
        """  
        start = time.time()  
        ori_pred = ori_pred if ori_pred != "." else ""  # Ensure the original prediction is not just a period  
    
        try:  
            if ori != tar and ori_pred != '':  # Proceed with translation only if languages are different and text is not empty  
                if self.translate_method == "google":  
                    # Adjust language codes for Google Translate  
                    ori = 'zh-TW' if ori == 'zh' else ori  
                    tar = 'zh-TW' if tar == 'zh' else tar  
                    translated_pred = self.google_translator.translate(ori_pred, src=ori, dest=tar).text  
                
                elif self.translate_method == "gpt-4o":  
                    try:  
                        translated_pred = self.gpt4o_translator.translate(ori_pred, ori, tar)  
                        if "403_Forbidden" in translated_pred:  
                            logger.error(f" | gpt-4o reject translate | use google translate to retry | ")  
                            # Retry translation using Google Translate if GPT-4o translation is forbidden  
                            ori = 'zh-TW' if ori == 'zh' else ori  
                            tar = 'zh-TW' if tar == 'zh' else tar  
                            translated_pred = self.google_translator.translate(ori_pred, src=ori, dest=tar).text  
                    except Exception as e:  
                        logger.error(f" | gpt-4o translate error: {e} | use google translate to retry | ")  
                        # Retry translation using Google Translate if an error occurs with GPT-4o  
                        ori = 'zh-TW' if ori == 'zh' else ori  
                        tar = 'zh-TW' if tar == 'zh' else tar  
                        translated_pred = self.google_translator.translate(ori_pred, src=ori, dest=tar).text  
                
                # elif self.translate_method == "gemma":  
                #     translated_pred = self.gemma_translator.translate(ori_pred, ori, tar)  
                
                elif self.translate_method in OLLAMA_MODEL:  
                    translated_pred = self.ollama_translator.chat(source_text=ori_pred, source_lang=ori, target_lang=tar)  
                
                else:  
                    translated_pred = ori_pred  # No translation needed if the method is not recognized  
            else:  
                translated_pred = ori_pred  # No translation needed if languages are the same or text is empty  
        
        except Exception as e:  
            translated_pred = ori_pred  # Fallback to original text in case of an error  
            logger.error(f" | translate() '{self.translate_method}' error: {e} | ")  
    
        end = time.time()  
        g_translate_time = end - start  # Calculate the time taken for translation  
    
        return translated_pred, g_translate_time, self.translate_method  
        
if __name__ == "__main__":  
    # argos  
    model = Model()  
    model.load_model("medium")  # Load the specified model by name  
    audio_file_path = "/mnt/audio/test.wav"  # Replace with the actual audio file path  
    ori = "en"  # Original language  
    tar = "ko"  # Target language  
    ori_pred, translated_pred, inference_time, g_translate_time = model.translate(audio_file_path, ori, tar)  
    print(f" | Original Transcription: {ori_pred} | ")  
    print(f" | Translated Transcription: {translated_pred} | ")  
    print(f" | Inference Time: {inference_time} seconds | ")  
    print(f" | Translation Time: {g_translate_time} seconds | ")  



