import logging  
import threading
import ctypes


logger = logging.getLogger(__name__)  

def translate_and_print(model, audio_file_path, ori, tar, result_queue, stop_event):  
    ori_pred, inference_time = model.transcribe(audio_file_path, ori)  
    translated_pred, g_translate_time, translate_method = model.translate(ori_pred, ori, tar)  
    result_queue.put((ori_pred, translated_pred, inference_time, g_translate_time, translate_method))  
    stop_event.set()  # Signal to stop the waiting thread  

def get_thread_id(thread):  
    if not thread.is_alive():  
        return None  
    for tid, tobj in threading._active.items():  
        if tobj is thread:  
            return tid  
    logger.debug(" | Could not determine the thread ID | ")
    raise AssertionError("Could not determine the thread ID")  

def stop_thread(thread):  
    thread_id = get_thread_id(thread)  
    if thread_id is not None:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))  
        if res == 0:  
            logger.debug(" | Invalid thread ID | ")
            raise ValueError("Invalid thread ID")  
        elif res != 1:  
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)  
            logger.debug(" | PyThreadState_SetAsyncExc failed | ")
            raise SystemError("PyThreadState_SetAsyncExc failed")  

def waiting_times(stop_event, times):  
    stop_event.wait(times)  # Wait for the event or timeout  
