import os  
import gc  
import time  
import torch
import whisper  
import logging  

from googletrans import Translator  
import argostranslate.translate  
from .gpt_translate import Gpt4oTranslate  

from lib.constant import ModlePath, OPTIONS
  
os.environ["ARGOS_DEVICE_TYPE"] = "cuda"  # Set ARGOS to use CUDA  
  
logger = logging.getLogger(__name__)  
  
class Model:  
    def __init__(self):  
        """  
        Initialize the Model class with default attributes.  
        """  
        self.model = None  
        self.model_version = None  
        self.models_path = ModlePath()  
        self.google_translator = Translator()  
        self.gpt4o_translator = Gpt4oTranslate()  
        self.translate_method = "argos"  
  
    def load_model(self, models_name):  
        """  
        Load the specified model based on the model's name.  
  
        :param models_name: str  
            The name of the model to be loaded.  
        :rtype: None  
        :logs: Loading status and time.  
        """  
        start = time.time()  
        try:  
            # Release old model resources  
            self._release_model() 
            self.model_version = models_name

            # Choose model weight  
            if models_name == "large_v2":  
                self.model = whisper.load_model(self.models_path.large_v2)  
            elif models_name == "medium":  
                self.model = whisper.load_model(self.models_path.medium)  
            elif models_name == "turbo":  
                self.model = whisper.load_model(self.models_path.turbo)  

            device = "cuda" if torch.cuda.is_available() else "cpu"  
            self.model.to(device)  
            end = time.time()  
            logger.info(f"Model '{models_name}' loaded in {end - start:.2f} seconds.")  
        except Exception as e:
            self.model_version = None
            logger.error(f'load_model() models_name:{models_name} error:{e}')
  
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
  
    def translate(self, audio_file_path, ori, tar):  
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
        result = self.model.transcribe(audio_file_path, **OPTIONS)  
        logger.debug(result)  
        ori_pred = result['text']  
        end = time.time()  
        inference_time = end - start  
        logger.debug(f"Inference time {inference_time} seconds.")  
  
        start = time.time()  
        if ori != tar and ori_pred != '' and self.translate_method == "google":  
            ori = 'zh-TW' if ori == 'zh' else ori  
            tar = 'zh-TW' if tar == 'zh' else tar  
            translated_pred = self.google_translator.translate(ori_pred, src=ori, dest=tar).text  
        elif ori != tar and ori_pred != '' and self.translate_method == "argos":  
            ori = 'zt' if ori == 'zh' else ori  
            tar = 'zt' if tar == 'zh' else tar  
            translated_pred = argostranslate.translate.translate(ori_pred, ori, tar)  
        elif ori != tar and ori_pred != '' and self.translate_method == "gpt-4o":  
            translated_pred = self.gpt4o_translator.translate(ori_pred, ori, tar)  
        else:  
            translated_pred = ori_pred  
        end = time.time()  
        g_translate_time = end - start  
  
        return ori_pred, translated_pred, inference_time, g_translate_time, self.translate_method  
  
if __name__ == "__main__":  
    # argos  
    model = Model()  
    model.load_model("medium")  # Load the specified model by name  
    audio_file_path = "/mnt/audio/test.wav"  # Replace with the actual audio file path  
    ori = "en"  # Original language  
    tar = "ko"  # Target language  
    ori_pred, translated_pred, inference_time, g_translate_time = model.translate(audio_file_path, ori, tar)  
    print(f"Original Transcription: {ori_pred}")  
    print(f"Translated Transcription: {translated_pred}")  
    print(f"Inference Time: {inference_time} seconds")  
    print(f"Translation Time: {g_translate_time} seconds")  