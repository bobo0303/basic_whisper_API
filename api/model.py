import os  
import gc  
import time  
import torch
import whisper  
import logging  

from googletrans import Translator  
from funasr import AutoModel  
from queue import Queue  

from api.gemma_translate import Gemma4BTranslate  
from api.ollama_translate import OllamaChat
from api.gpt_translate import Gpt4oTranslate  

from api.text_postprocess import extract_sensevoice_result_text
from lib.constant import ModlePath, OPTIONS, SENSEVOCIE_PARMATER, IS_PUNC, PUNC_PARMATER, GEMMA_12B_QAT_CONFIG
  
  
logger = logging.getLogger(__name__)  
  
class Model:  
    def __init__(self):  
        """  
        Initialize the Model class with default attributes.  
        """  
        self.models_path = ModlePath()  
        self.gemma_translator = Gemma4BTranslate()
        self.ollama_translator = OllamaChat(GEMMA_12B_QAT_CONFIG)
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
                    logger.info(f" | Model \'ct-punc\' loaded in {end - start:.2f} seconds. | ")
            end = time.time()  
            logger.info(f" | Model '{models_name}' loaded in {end - start:.2f} seconds. | ")  
        except Exception as e:
            self.model_version = None
            logger.error(f' | load_model() models_name: {models_name} error: {e} | ')
  
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
            logger.info(" | Previous model resources have been released. | ")  
  
    def change_translate_method(self, method_name):  
        """  
        Change the translation method used by the model.  
  
        :param method_name: str  
            The name of the translation method to be used.  
        :rtype: None  
        """  
        self.translate_method = method_name  
  
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
        if self.model_version == "sensevoice":
            result = self.model.generate(audio_file_path, **OPTIONS)
            ori_pred = result[0]['text']
            if IS_PUNC:
                ori_pred = self.punc_model.generate(input=ori_pred)
                ori_pred = ori_pred[0]['text']
            ori_pred = extract_sensevoice_result_text(ori_pred.lower())
        else:
            result = self.model.transcribe(audio_file_path, **OPTIONS)  
            logger.debug(result)  
            ori_pred = result['text']  
        end = time.time()  
        inference_time = end - start  
        logger.debug(f" | Inference time {inference_time} seconds. | ")  
  
        return ori_pred, inference_time
  
    def translate(self, ori_pred, ori, tar):
        start = time.time()  
        try:
            if ori != tar and ori_pred != '':
                if self.translate_method == "google":  
                    ori = 'zh-TW' if ori == 'zh' else ori  
                    tar = 'zh-TW' if tar == 'zh' else tar  
                    translated_pred = self.google_translator.translate(ori_pred, src=ori, dest=tar).text  
                elif self.translate_method == "gpt-4o": 
                    try:
                        translated_pred = self.gpt4o_translator.translate(ori_pred, ori, tar)  
                        if "403_Forbidden" in translated_pred:
                            ori = 'zh-TW' if ori == 'zh' else ori  
                            tar = 'zh-TW' if tar == 'zh' else tar  
                            translated_pred = self.google_translator.translate(ori_pred, src=ori, dest=tar).text  
                    except Exception as e:
                        logger.error(f" | gpt-4o translate error: {e} | use google translate to retry | ")
                        ori = 'zh-TW' if ori == 'zh' else ori  
                        tar = 'zh-TW' if tar == 'zh' else tar  
                        translated_pred = self.google_translator.translate(ori_pred, src=ori, dest=tar).text  
                elif self.translate_method == "gemma":  
                    translated_pred = self.gemma_translator.translate(ori_pred, ori, tar)   
                elif self.translate_method == "ollama":
                    translated_pred = self.ollama_translator.chat(sourse_text=ori_pred, sourse_lang=ori, target_lang=tar)  
            else:  
                translated_pred = ori_pred
        except Exception as e:
            translated_pred = ori_pred
            logger.error(f" | translate() '{self.translate_method}' error: {e} | ") 
        end = time.time()  
        g_translate_time = end - start  
        
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


# if __name__ == "__main__":  
#     import threading  
#     import ctypes  
#     import time  
    
#     def translate_and_print(model, audio_file_path, ori, tar):  
#         print("thread 2 start")
#         ori_pred, translated_pred, inference_time, g_translate_time, translate_method = model.translate(audio_file_path, ori, tar)  
#         print(f"2Original Transcription: {ori_pred}")  
#         print(f"2Translated Transcription: {translated_pred}")  
#         print(f"2Inference Time: {inference_time} seconds")  
#         print(f"2Translation Time: {g_translate_time} seconds") 
#         print("thread 2 end")

#         return ori_pred, translated_pred, inference_time, g_translate_time, translate_method  
        

#     def get_thread_id(thread):  
#         if not thread.is_alive():  
#             raise threading.ThreadError("The thread is not active")  
#         for tid, tobj in threading._active.items():  
#             if tobj is thread:  
#                 return tid  
#         raise AssertionError("Could not determine the thread ID")  
    
#     def stop_thread(thread):  
#         thread_id = get_thread_id(thread)  
#         res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))  
#         if res == 0:  
#             raise ValueError("Invalid thread ID")  
#         elif res != 1:  
#             ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)  
#             raise SystemError("PyThreadState_SetAsyncExc failed")  
        
#     model = Model()  
#     model.load_model("medium")  # Load the specified model by name  
  
#     audio_file_path_1 = "/mnt/audio/123.wav"  # Replace with the actual audio file path  
#     audio_file_path_2 = "/mnt/audio/test.wav"  # Replace with the actual audio file path  
#     ori = "en"  # Original language  
#     tar = "ko"  # Target language  
  
#     def times():  
#         print("Thread 1 is running")  
#         time.sleep(0.5)  # Simulate some work in thread 1  
#         print("Thread 1 is done")  
  
#     # Create two threads  
#     thread1 = threading.Thread(target=times)  
#     thread2 = threading.Thread(target=translate_and_print, args=(model, audio_file_path_2, ori, tar))  
  
#     # Start the threads  
#     thread1.start()  
#     thread2.start()  
  
#     # Wait for thread 1 to complete  
#     thread1.join()  
      
#     # Forcefully stop thread 2  
#     stop_thread(thread2) 

#     ori_pred, translated_pred, inference_time, g_translate_time, _ = model.translate(audio_file_path_1, ori, tar)  
#     print(f"Original Transcription: {ori_pred}")  
#     print(f"Translated Transcription: {translated_pred}")  
#     print(f"Inference Time: {inference_time} seconds")  
#     print(f"Translation Time: {g_translate_time} seconds") 

