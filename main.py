from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
import os  
import time  
import pytz  
import logging  
import logging.handlers  
import uvicorn  
import datetime  
import requests  
import threading 
from queue import Queue  
from threading import Thread, Event  
from api.model import Models  
from api.threading_api import translate_and_print, waiting_times, stop_thread
from lib.data_object import LoadModelRequest, LoadMethodRequest  
from lib.base_object import BaseResponse  
from lib.constant import ResponseSTT, TranscriptionData, VSTResponseSTT, TextData, WAITING_TIME, LANGUAGE_LIST, TRANSCRIBE_METHODS, TRANSLATE_METHODS  
  
#############################################################################  
  
if not os.path.exists("./audio"):  
    os.mkdir("./audio")  
if not os.path.exists("./logs"):  
    os.mkdir("./logs")  
    
# Configure logging  
log_format = "%(asctime)s - %(message)s"  
log_file = "logs/app.log"  
  
# Create a logger  
logger = logging.getLogger(__name__)  
  
# Avoid adding handlers multiple times  
if not logger.hasHandlers():  
    logger.setLevel(logging.INFO)  
  
    # Create a file handler for logging to a file  
    file_handler = logging.handlers.RotatingFileHandler(  
        log_file, maxBytes=10*1024*1024, backupCount=5  
    )  
    file_handler.setFormatter(logging.Formatter(log_format))  
  
    # Create a stream handler for logging to console  
    stream_handler = logging.StreamHandler()  
    stream_handler.setFormatter(logging.Formatter(log_format))  
  
    # Add handlers to the logger  
    logger.addHandler(file_handler)  
    logger.addHandler(stream_handler)  
  
# Configure UTC+8 time  
utc_now = datetime.datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  

app = FastAPI()  
model = Models()  
queue = Queue()  
  
@app.get("/")  
def HelloWorld(name:str=None):  
    return {"Hello": f"World {name}"}  

##############################################################################  
  
@app.on_event("startup")  
async def load_default_model_preheat():  
    """  
    The process of loading the default model and preheating on startup.  
  
    This function loads the default model and preheats it by running a few  
    inference operations. This is useful to reduce the initial latency  
    when the model is first used.  
  
    :param None: The function does not take any parameters.  
    :rtype: None: The function does not return any value.  
    :logs: Loading and preheating status and times.  
    """  
    logger.info("#####################################################")  
    logger.info(f"Start to loading default model.")  
    # load model  
    default_model = "sensevoice"  
    model.load_model(default_model)  # Directly load the default model  
    logger.info(f"Default model {default_model} has been loaded successfully.")  
    # preheat  
    logger.info(f"Start to preheat model.")  
    default_audio = "audio/test.wav"  
    start = time.time()  
    for _ in range(5):  
        model.transcribe(default_audio, "en")  
    end = time.time()  
      
    logger.info(f"Preheat model has been completed in {end - start:.2f} seconds.")  
    logger.info("#####################################################")  
    delete_old_audio_files()  

@app.get("/get_current_models")  
async def get_items():  
    logger.info("################# inference model #############################")  
    logger.info(f"current ASR model is {model.model_version}")  
    logger.info(f"current Translate model is {model.translate_method}")  
    logger.info("###############################################################")  
    return BaseResponse(message=f" | current ASR model is \"{model.model_version}\" | current Translate model is \"{model.translate_method}\" | ", data=model.model_version)  

@app.get("/list_optional_items")  
async def get_items():  
    """  
    List the optional items for inference models and translate methods.  
  
    This endpoint provides information about the available inference models  
    and translation methods that can be selected.  
  
    :rtype: str: A string listing the available inference models and translation methods.  
    """  
    logger.info("################# inference model #############################")  
    logger.info(f"You can choose {TRANSCRIBE_METHODS}")  
    logger.info("################# translate methods ###########################")  
    logger.info(f"You can choose {TRANSLATE_METHODS}")  
    logger.info("###############################################################")  
    return BaseResponse(message=f" | inference model: You can choose {TRANSCRIBE_METHODS} | translate method: You can choose {TRANSLATE_METHODS} | ", data=None)  
  
@app.post("/change_translate_method")  
async def change_translate_method(request: LoadMethodRequest):
    """  
    Change the translation method.  
  
    This endpoint allows the user to change the translation method used  
    by the model.  
  
    :param request: LoadMethodRequest  
        The request object containing the new translation method's name.  
    :rtype: BaseResponse  
        A response indicating the success or failure of changing the translation method.  
    """  
    method_name = request.method_name.lower()  
    if method_name in TRANSLATE_METHODS:  
        model.change_translate_method(method_name)  
        logger.info(f"Translate method {method_name} has been changed successfully.")  
        return BaseResponse(message=f"Translate method {method_name} has been changed successfully.", data=None)  
    else:
        return BaseResponse(status="FAILED", message=f"Translate method {method_name} not in allow methods \"{TRANSLATE_METHODS}\".", data=None)
  
@app.post("/change_transcribe_method")  
async def load_model(request: LoadModelRequest):  
    """  
    Change the transcribe method.  
  
    This endpoint allows the user to load a specified model for inference.  
  
    :param request: LoadModelRequest  
        The request object containing the model's name to be loaded.  
    :rtype: BaseResponse  
        A response indicating the success or failure of the model loading process.  
    """  
    models_name = request.models_name.lower()  
    if models_name in TRANSCRIBE_METHODS:  
        model.load_model(models_name)  
        logger.info(f"Model {request.models_name} has been loaded successfully.")  
        return BaseResponse(message=f"Model {request.models_name} has been loaded successfully.", data=None)  
    else:
        return BaseResponse(status="FAILED", message=f"Transcribe method {models_name} not in allow methods \"{TRANSCRIBE_METHODS}\".", data=None)
  
@app.post("/translate")  
async def translate(  
    file: UploadFile = File(...),  
    transcription_request: TranscriptionData = Depends()  
):  
    """  
    Transcribe and translate an audio file.  
  
    This endpoint receives an audio file and its associated metadata, and  
    performs transcription and translation on the audio file.  
  
    :param file: UploadFile  
        The audio file to be transcribed.  
    :param meeting_id: str  
        The ID of the meeting.  
    :param device_id: str  
        The ID of the device.  
    :param audio_uid: str  
        The unique ID of the audio.  
    :param time: str  
        The start time of the audio.  
    :param o_lang: str  
        The original language of the audio.  
    :param t_lang: str  
        The target language for translation.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    
    # Extract data from transcription_request  
    times = str(transcription_request.times)
    o_lang = transcription_request.o_lang.lower()
    t_lang = transcription_request.t_lang.lower()

    response_data = ResponseSTT(  
        meeting_id=transcription_request.meeting_id,  
        device_id=transcription_request.device_id,  
        ori_lang=o_lang,  
        ori_text="",  
        trans_lang=t_lang,  
        trans_text="",  
        times=times,  
        audio_uid=transcription_request.audio_uid, 
        transcribe_time=0.0,
        translate_time=0.0, 
    )  
    
    file_name = times + ".wav"  
    audio_buffer = f"audio/{file_name}"  
    with open(audio_buffer, 'wb') as f:  
        f.write(file.file.read())  
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status="FAILED", message="The audio file does not exist, please check the audio path.", data=response_data)  

    if model.model_version is None:
        return BaseResponse(status="FAILED", message="model haven't been load successfull. may out of memory please check again", data=response_data)  
      
    if o_lang not in LANGUAGE_LIST or t_lang not in LANGUAGE_LIST:  
        logger.info(f"One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}.")  
        return BaseResponse(status="FAILED", message=f"One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}.", data=response_data)  
    
    try:
        # Create a queue to hold the return value
        result_queue = Queue()
        # Create an event to signal stopping  
        stop_event = threading.Event()  

        # Create timing thread and inference thread
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, WAITING_TIME))  
        inference_thread = threading.Thread(target=translate_and_print, args=(model, audio_buffer, o_lang, t_lang, result_queue, stop_event))

        # Start the threads
        time_thread.start()
        inference_thread.start()

        # Wait for timing thread to complete check if the inference thread is active to close
        time_thread.join()
        stop_thread(inference_thread) 

        os.remove(audio_buffer)  

        # Get the result from the queue  
        if not result_queue.empty():  
            o_result, t_result, inference_time, g_translate_time, translate_method = result_queue.get() 
            response_data.ori_text = o_result
            response_data.trans_text = t_result
            response_data.transcribe_time = inference_time
            response_data.translate_time = g_translate_time
            logger.debug(response_data.model_dump_json())  
            logger.info(f" | device_id: {response_data.device_id} | audio_uid: {response_data.audio_uid} | language: {o_lang} -> {t_lang} | translate_method: {translate_method} |")  
            logger.info(f" | transcription: {response_data.ori_text} |")  
            logger.info(f" | translation: {response_data.trans_text} |")  
            logger.info(f" | inference has been completed in {inference_time:.2f} seconds. | translate has been completed in {g_translate_time:.2f} seconds.")  
            state="OK"
        else:
            logger.info(f" | Inference has exceeded the upper limit time and has been stopped |")  
            state="FAILED"

        return BaseResponse(status=state, message=f" | transcription: {response_data.ori_text} | translation: {response_data.trans_text} | ", data=response_data)  
    except Exception as e:
        logger.error(f'iference() error:{e}')
        return BaseResponse(status="FAILED", message=f" | iference() error:{e} | ", data=response_data)  

@app.post("/text_translate")  
async def text_translate(  
    translate_request: TextData = Depends()
):  
    o_lang = translate_request.o_lang.lower()
    t_lang = translate_request.t_lang.lower()
    o_result = translate_request.ori_text
    
    response_data = VSTResponseSTT(
        ori_text="",
        tar_text="",
        )
    
    try:
        translated_pred, g_translate_time, translate_method = model.translate(o_result, o_lang, t_lang)
        response_data.ori_text = o_result
        response_data.tar_text = translated_pred
        logger.info(f" | language: {o_lang} -> {t_lang} | translate_method: {translate_method} | translate has been completed in {g_translate_time:.2f} seconds. |")  
        logger.info(f" | transcription: {o_result} |")  
        logger.info(f" | translation: {translated_pred} |") 
        state="OK"
        return BaseResponse(status=state, message=f" | input text: {o_result} | translation: {translated_pred} | ", data=response_data)  
    except Exception as e:
        logger.error(f'iference() error:{e}')
        translated_pred = o_result
        state="FAILED"
        return BaseResponse(status=state, message=f" | iference() error:{e} | ", data=response_data)  


# Clean up audio files  
def delete_old_audio_files():  
    """  
    The process of deleting old audio files  
    :param  
    ----------  
    None: The function does not take any parameters  
    :rtype  
    ----------  
    None: The function does not return any value  
    :logs  
    ----------  
    Deleted old files  
    """  
    current_time = time.time()  
    audio_dir = "./audio"  
    for filename in os.listdir(audio_dir):  
        if filename == "test.wav":  # Skip specific file  
            continue  
        file_path = os.path.join(audio_dir, filename)  
        if os.path.isfile(file_path):  
            file_creation_time = os.path.getctime(file_path)  
            # Delete files older than a day  
            if current_time - file_creation_time > 24 * 60 * 60:  
                os.remove(file_path)  
                logger.info(f"Deleted old file: {file_path}")  
  
# Daily task scheduling  
def schedule_daily_task(stop_event):  
    while not stop_event.is_set():  
        if local_now.hour == 0 and local_now.minute == 0:  
            delete_old_audio_files()  
            time.sleep(60)  # Prevent triggering multiple times within the same minute  
        time.sleep(1)  
  
# Start daily task scheduling  
stop_event = Event()  
task_thread = Thread(target=schedule_daily_task, args=(stop_event,))  
task_thread.start()  
  
@app.on_event("shutdown")  
def shutdown_event():  
    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  
  
if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"  
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)  
    
    
 

