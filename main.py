from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Form, Depends
from fastapi.responses import StreamingResponse  
import os  
import time  
import pytz  
import asyncio  
import logging  
import uvicorn  
import datetime  
import threading 
from queue import Queue  
from threading import Thread, Event  
from api.model import Model  
from api.threading_api import translate_and_print, ws_translate_and_print, waiting_times, stop_thread
from lib.base_object import BaseResponse  
from lib.constant import ResponseSTT, LoadModelRequest, LoadMethodRequest, TranscriptionData, VSTTranscriptionData, VSTResponseSTT, TextData, WAITING_TIME, LANGUAGE_LIST, ASR_METHODS, TRANSLATE_METHODS  
  
#############################################################################  
  
if not os.path.exists("./audio"):  
    os.mkdir("./audio")  
if not os.path.exists("./logs"):  
    os.mkdir("./logs")  
    
# Configure logging  
log_format = "%(asctime)s - %(message)s"  # Output timestamp and message content  
log_file = "logs/app.log"  
logging.basicConfig(level=logging.INFO, format=log_format)  
  
# Create a file handler  
file_handler = logging.handlers.RotatingFileHandler(  
    log_file, maxBytes=10*1024*1024, backupCount=5  
)  
file_handler.setFormatter(logging.Formatter(log_format))  
  
# Create a console handler  
console_handler = logging.StreamHandler()  
console_handler.setFormatter(logging.Formatter(log_format))  
  
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)  # Ensure logger level is set to INFO or lower  
  
# Clear existing handlers to prevent duplicate logs  
if not logger.handlers:  
    logger.addHandler(file_handler)  
    logger.addHandler(console_handler)  # Add console handler 

logger.propagate = False  
  
# Configure UTC+8 time  
utc_now = datetime.datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  
  
app = FastAPI()  
model = Model()  
queue = Queue()  
waiting_list = []

# Global event to control SSE connection  
sse_stop_event = Event()  
  
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
    logger.info(f" | ##################################################### | ")  
    logger.info(f" | Start to loading default model. | ")  
    # load model  
    default_model = "large_v2"  
    model.load_model(default_model)  # Directly load the default model  
    logger.info(f" | Default model {default_model} has been loaded successfully. | ")  
    # preheat  
    logger.info(f" | Start to preheat model. | ")  
    default_audio = "audio/test.wav"  
    start = time.time()  
    for _ in range(5):  
        model.transcribe(default_audio, "en")  
    end = time.time()  
    logger.info(f" | Preheat model has been completed in {end - start:.2f} seconds. | ")  
    logger.info(f" | ##################################################### | ")  
    delete_old_audio_files()  

@app.get("/get_current_ASR_model")  
async def get_items():  
    logger.info(f" | ################# inference model ############################# | ")  
    logger.info(f" | current ASR model is {model.model_version} ")  
    logger.info(f" | ############################################################### | ")  
    return BaseResponse(message=f" | current ASR model is {model.model_version} | ", data=model.model_version)  

@app.get("/list_optional_items")  
async def get_items():  
    """  
    List the optional items for inference models and translate methods.  
  
    This endpoint provides information about the available inference models  
    and translation methods that can be selected.  
  
    :rtype: str: A string listing the available inference models and translation methods.  
    """  
    logger.info(f" | ################# inference model ############################# | ")  
    logger.info(f" | You can choose {ASR_METHODS} | ")  
    logger.info(f" | ################# translate methods ########################### | ")  
    logger.info(f" | You can choose {TRANSLATE_METHODS} | ")  
    logger.info(f" | ############################################################### | ")  
    return BaseResponse(message=f" | inference model: You can choose {ASR_METHODS} | translate method: You can choose {TRANSLATE_METHODS} | ", data=None)  
  
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
    # Convert the method name to lowercase  
    method_name = request.method_name.lower()  
      
    # Check if the method name is in the list of supported translation methods  
    if method_name in TRANSLATE_METHODS:  
        # Change the translation method  
        model.change_translate_method(method_name)  
        logger.info(f" | Translate method '{method_name}' has been changed successfully. | ")  
      
    # Return a response indicating the success of the operation  
    return BaseResponse(message=f" | Translate method '{method_name}' has been changed successfully. | ", data=None)  
  
@app.post("/load_model")  
async def load_model(request: LoadModelRequest):  
    """  
    Load a specified model.  
      
    This endpoint allows the user to load a specified model for inference.  
      
    :param request: LoadModelRequest  
        The request object containing the model's name to be loaded.  
    :rtype: BaseResponse  
        A response indicating the success or failure of the model loading process.  
    """  
    # Convert the model's name to lowercase  
    models_name = request.models_name.lower()  
      
    # Check if the model's name exists in the model's path  
    if not hasattr(model.models_path, models_name):  
        # Raise an HTTPException if the model is not found  
        raise HTTPException(status_code=400, detail="Model not found")  
      
    # Load the specified model  
    model.load_model(models_name)  
    logger.info(f" | Model {request.models_name} has been loaded successfully. | ")  
      
    # Return a response indicating the success of the model loading process  
    return BaseResponse(message=f" | Model {request.models_name} has been loaded successfully. | ", data=None)  
  
  
@app.post("/rtt_translate", description="**[DEPRECATED]** This endpoint is deprecated and will be removed in the future. Please use `/rtt_translate/v2` instead.")  
async def rtt_translate(  
    file: UploadFile = File(...),  
    transcription_request: TranscriptionData = Depends()  
):  
    """  
    Transcribe and translate an audio file.  
      
    This endpoint receives an audio file and its associated metadata, and  
    performs transcription and translation on the audio file.  
      
    :param file: UploadFile  
        The audio file to be transcribed.  
    :param transcription_request: TranscriptionData  
        The request object containing metadata of the audio file, such as:  
        - meeting_id: str  
            The ID of the meeting.  
        - device_id: str  
            The ID of the device.  
        - audio_uid: str  
            The unique ID of the audio.  
        - times: datetime  
            The start time of the audio.  
        - o_lang: str  
            The original language of the audio.  
        - t_lang: str  
            The target language for translation.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    # Extract data from transcription_request  
    times = str(transcription_request.times)  
    o_lang = transcription_request.o_lang.lower()  
    t_lang = transcription_request.t_lang.lower()  
  
    # Create response data structure  
    response_data = ResponseSTT(  
        meeting_id=transcription_request.meeting_id,  
        device_id=transcription_request.device_id,  
        ori_lang=o_lang,  
        ori_text="",  
        trans_lang=t_lang,  
        trans_text="",  
        times=str(times),  
        audio_uid=transcription_request.audio_uid,  
        transcribe_time=0.0,  
        translate_time=0.0,  
    )  
  
    # Save the uploaded audio file  
    file_name = times + ".wav"  
    audio_buffer = f"audio/{file_name}"  
    with open(audio_buffer, 'wb') as f:  
        f.write(file.file.read())  
  
    # Check if the audio file exists  
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status="FAILED", message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
  
    # Check if the model has been loaded  
    if model.model_version is None:  
        return BaseResponse(status="FAILED", message=" | model haven't been load successful. may be out of memory please check again | ", data=response_data)  
  
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST or t_lang not in LANGUAGE_LIST:  
        logger.info(f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status="FAILED", message=f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, WAITING_TIME))  
        inference_thread = threading.Thread(target=translate_and_print, args=(model, audio_buffer, result_queue, o_lang, t_lang, stop_event))  
  
        # Start the threads  
        time_thread.start()  
        inference_thread.start()  
  
        # Wait for timing thread to complete and check if the inference thread is active to close  
        time_thread.join()  
        stop_thread(inference_thread)  
  
        # Remove the audio buffer file
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
            logger.info(f" | inference has been completed in {inference_time:.2f} seconds. | translation has been completed in {g_translate_time:.2f} seconds. |")  
            state = "OK"  
        else:  
            logger.info(f" | Inference has exceeded the upper limit time and has been stopped |")  
            state = "FAILED"  
  
        return BaseResponse(status=state, message=f" | transcription: {response_data.ori_text} | translation: {response_data.trans_text} | ", data=response_data)  
    except Exception as e:  
        logger.error(f'inference() error: {e}')  
        return BaseResponse(status="FAILED", message=f" | inference() error: {e} | ", data=response_data)  
    
    
@app.post("/rtt_translate/v2")  
async def rtt_translate(  
    file: UploadFile = File(...),  
    meeting_id: str = Form(...),  
    device_id: str = Form(...),  
    audio_uid: str = Form(...),  
    times: datetime.datetime = Form(...),  
    o_lang: str = Form(...),  
    t_lang: str = Form(...)  
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
    :param times: datetime.datetime  
        The start time of the audio.  
    :param o_lang: str  
        The original language of the audio.  
    :param t_lang: str  
        The target language for translation.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    # Convert times to string format  
    times = str(times)  
    # Convert original language and target language to lowercase  
    o_lang = o_lang.lower()  
    t_lang = t_lang.lower()  
  
    # Create response data structure  
    response_data = ResponseSTT(  
        meeting_id=meeting_id,  
        device_id=device_id,  
        ori_lang=o_lang,  
        ori_text="",  
        trans_lang=t_lang,  
        trans_text="",  
        times=str(times),  
        audio_uid=audio_uid,  
        transcribe_time=0.0,  
        translate_time=0.0,  
    )  
  
    # Save the uploaded audio file  
    file_name = times + ".wav"  
    audio_buffer = f"audio/{file_name}"  
    with open(audio_buffer, 'wb') as f:  
        f.write(file.file.read())  
  
    # Check if the audio file exists  
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status="FAILED", message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
  
    # Check if the model has been loaded  
    if model.model_version is None:  
        return BaseResponse(status="FAILED", message=" | model haven't been load successfully. may out of memory please check again | ", data=response_data)  
  
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST or t_lang not in LANGUAGE_LIST:  
        logger.info(f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status="FAILED", message=f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, WAITING_TIME))  
        inference_thread = threading.Thread(target=translate_and_print, args=(model, audio_buffer, result_queue, o_lang, t_lang, stop_event))  
  
        # Start the threads  
        time_thread.start()  
        inference_thread.start()  
  
        # Wait for timing thread to complete and check if the inference thread is active to close  
        time_thread.join()  
        stop_thread(inference_thread)  
  
        # Remove the audio buffer file  
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
            logger.info(f" | inference has been completed in {inference_time:.2f} seconds. | translate has been completed in {g_translate_time:.2f} seconds. |")  
            state = "OK"  
        else:  
            logger.info(f" | Inference has exceeded the upper limit time and has been stopped |")  
            state = "FAILED"  
  
        return BaseResponse(status=state, message=f" | transcription: {response_data.ori_text} | translation: {response_data.trans_text} | ", data=response_data)  
    except Exception as e:  
        logger.error(f" | inference() error: {e} | ")  
        return BaseResponse(status="FAILED", message=f" | inference() error: {e} | ", data=response_data)  
    
@app.websocket("/ws/rtt_translate")  
async def websocket_endpoint(websocket: WebSocket):  
    """  
    WebSocket endpoint for real-time transcription and translation.  
      
    This endpoint allows clients to send audio files over WebSocket for real-time  
    transcription and translation.  
      
    :param websocket: WebSocket  
        The WebSocket connection to the client.  
    """  
    await websocket.accept()  
    waiting_list = []  
  
    # Clear the result queue  
    while not model.result_queue.empty():  
        model.result_queue.get()  
  
    async def process_audio():  
        """  
        Process audio files from the waiting list for transcription and translation.  
          
        This function runs in a separate asyncio task and continuously processes  
        audio files from the waiting list.  
        """  
        while True:  
            if waiting_list and not model.processing:  
                response_data = waiting_list[0].model_copy()  
                waiting_list.pop(0)  
                audio_buffer = f"audio/{response_data.times}.wav"  
                o_lang = response_data.ori_lang  
                t_lang = response_data.trans_lang  
  
                try:  
                    # Create an event to signal stopping  
                    stop_event = threading.Event()  
                    # Create timing thread and inference thread  
                    time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, WAITING_TIME))  
                    inference_thread = threading.Thread(target=ws_translate_and_print, args=(model, audio_buffer, o_lang, t_lang, stop_event))  
                      
                    # Start the threads  
                    time_thread.start()  
                    inference_thread.start()  
                      
                    # Wait for timing thread to complete and stop the inference thread if still running  
                    time_thread.join()  
                    stop_thread(inference_thread)  
                      
                    # Remove the audio buffer file  
                    os.remove(audio_buffer)  
                      
                    # Process results from the result queue  
                    while not model.result_queue.empty():  
                        o_result, t_result, inference_time, g_translate_time, translate_method = model.result_queue.get()  
                        response_data.ori_text = o_result  
                        response_data.trans_text = t_result  
                        response_data.transcribe_time = inference_time  
                        response_data.translate_time = g_translate_time  
                          
                        logger.debug(response_data.model_dump_json())  
                        logger.info(f" | device_id: {response_data.device_id} | audio_uid: {response_data.audio_uid} | language: {response_data.ori_lang} -> {response_data.trans_lang} | translate_method: {translate_method} | ")  
                        logger.info(f" | transcription: {response_data.ori_text} | ")  
                        logger.info(f" | translation: {response_data.trans_text} | ")  
                        logger.info(f" | Inference completed in {inference_time:.2f} seconds. Translation completed in {g_translate_time:.2f} seconds. | ")  
                          
                        await websocket.send_json(BaseResponse(status="OK", message=f" | transcription: {response_data.ori_text} | translation: {response_data.trans_text} | ", data=response_data).model_dump())  
                except Exception as e:  
                    logger.error(f' | inference() error: {e} | ')  
            await asyncio.sleep(0.1)  
  
    # Create a background task to process audio files  
    asyncio.create_task(process_audio())  
  
    try:  
        while True:  
            # Receive JSON data from the client  
            data = await websocket.receive_json()  
            transcription_request = TranscriptionData(**data)  
            # Receive audio bytes from the client  
            audio_bytes = await websocket.receive_bytes()  
  
            # Create a response data structure  
            response_data = ResponseSTT(  
                meeting_id=transcription_request.meeting_id,  
                device_id=transcription_request.device_id,  
                ori_lang=transcription_request.o_lang.lower(),  
                ori_text="",  
                trans_lang=transcription_request.t_lang.lower(),  
                trans_text="",  
                times=str(transcription_request.times),  
                audio_uid=transcription_request.audio_uid,  
                transcribe_time=0.0,  
                translate_time=0.0,  
            )  
  
            previous_waiting_list = waiting_list.copy()  
            audio_uid_exist = False  
  
            # Check if the audio UID already exists in the waiting list  
            for item in previous_waiting_list:  
                if item.audio_uid == response_data.audio_uid:  
                    audio_uid_exist = True  
                    if item.times < response_data.times:  
                        waiting_list.remove(item)  
                        waiting_list.append(response_data)  
                        audio = f"audio/{item.times}.wav"  
                        if os.path.exists(audio):  
                            os.remove(audio)  
                    break  
  
            if not audio_uid_exist:  
                waiting_list.append(response_data)  
  
            if previous_waiting_list != waiting_list:  
                file_name = f"{response_data.times}.wav"  
                audio_buffer = f"audio/{file_name}"  
                with open(audio_buffer, 'wb') as f:  
                    f.write(audio_bytes)  
    except WebSocketDisconnect:  
        logger.info(" | Client disconnected | ")  
        
@app.post("/sse_rtt_translate", description="**[DEPRECATED]** This endpoint is deprecated and will be removed in the future. Please use `/sse_rtt_translate/v2` instead.")  
async def sse_rtt_translate(  
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
    times = transcription_request.times.isoformat()  
    o_lang = transcription_request.o_lang.lower()  
    t_lang = transcription_request.t_lang.lower()  
      
    # Create response data structure  
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
      
    try:  
        previous_waiting_list = waiting_list.copy()  
        audio_uid_exist = False  
          
        # Check if the audio UID already exists in the waiting list  
        for item in previous_waiting_list:  
            if item.audio_uid == response_data.audio_uid:  
                audio_uid_exist = True  
                if item.times < response_data.times:  
                    waiting_list.remove(item)  
                    waiting_list.append(response_data)  
                    audio = f"audio/{item.times}.wav"  
                    if os.path.exists(audio):  
                        os.remove(audio)  
                break  
          
        if not audio_uid_exist:  
            waiting_list.append(response_data)  
          
        if previous_waiting_list != waiting_list:  
            file_name = f"{response_data.times}.wav"  
            audio_buffer = f"audio/{file_name}"  
            with open(audio_buffer, 'wb') as f:  
                f.write(file.file.read())  
          
        # Check if the audio file exists  
        if not os.path.exists(audio_buffer):  
            return BaseResponse(status="FAILED", message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
          
        # Check if the model has been loaded  
        if model.model_version is None:  
            return BaseResponse(status="FAILED", message=" | model haven't been load successful. may out of memory please check again | ", data=response_data)  
          
        # Check if the languages are in the supported language list  
        if o_lang not in LANGUAGE_LIST or t_lang not in LANGUAGE_LIST:  
            logger.info(f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
            return BaseResponse(status="FAILED", message=f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
          
        return BaseResponse(status="OK", message=" | Request added to the waiting list. | ", data=None)  
    except Exception as e:  
        logger.error(f' | save info error: {e} | ')  
        return BaseResponse(status="FAILED", message=f" | save info error: {e} | ", data=response_data)  
    
@app.post("/sse_rtt_translate/v2")  
async def sse_rtt_translate(  
    file: UploadFile = File(...),  
    meeting_id: str = Form(...),  
    device_id: str = Form(...),  
    audio_uid: str = Form(...),  
    times: datetime.datetime = Form(...),  
    o_lang: str = Form(...),  
    t_lang: str = Form(...)  
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
    :param times: datetime.datetime  
        The start time of the audio.  
    :param o_lang: str  
        The original language of the audio.  
    :param t_lang: str  
        The target language for translation.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    # Extract data from transcription_request  
    times = times.isoformat()  
    o_lang = o_lang.lower()  
    t_lang = t_lang.lower()  
      
    # Create response data structure  
    response_data = ResponseSTT(  
        meeting_id=meeting_id,  
        device_id=device_id,  
        ori_lang=o_lang,  
        ori_text="",  
        trans_lang=t_lang,  
        trans_text="",  
        times=times,  
        audio_uid=audio_uid,  
        transcribe_time=0.0,  
        translate_time=0.0,  
    )  
      
    try:  
        previous_waiting_list = waiting_list.copy()  
        audio_uid_exist = False  
          
        # Check if the audio UID already exists in the waiting list  
        for item in previous_waiting_list:  
            if item.audio_uid == response_data.audio_uid:  
                audio_uid_exist = True  
                if item.times < response_data.times:  
                    waiting_list.remove(item)  
                    waiting_list.append(response_data)  
                    audio = f"audio/{item.times}.wav"  
                    if os.path.exists(audio):  
                        os.remove(audio)  
                break  
          
        if not audio_uid_exist:  
            waiting_list.append(response_data)  
          
        if previous_waiting_list != waiting_list:  
            file_name = f"{response_data.times}.wav"  
            audio_buffer = f"audio/{file_name}"  
            with open(audio_buffer, 'wb') as f:  
                f.write(file.file.read())  
          
        # Check if the audio file exists  
        if not os.path.exists(audio_buffer):  
            return BaseResponse(status="FAILED", message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
          
        # Check if the model has been loaded  
        if model.model_version is None:  
            return BaseResponse(status="FAILED", message=" | model haven't been load successful. may out of memory please check again | ", data=response_data)  
          
        # Check if the languages are in the supported language list  
        if o_lang not in LANGUAGE_LIST or t_lang not in LANGUAGE_LIST:  
            logger.info(f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
            return BaseResponse(status="FAILED", message=f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
          
        return BaseResponse(status="OK", message=" | Request added to the waiting list. | ", data=None)  
    except Exception as e:  
        logger.error(f' | save info error: {e} | ')  
        return BaseResponse(status="FAILED", message=f" | save info error: {e} | ", data=response_data)  
    
  
@app.get("/sse_rtt_translate")  
async def sse_rtt_translate():  
    """  
    Server-Sent Events endpoint to handle real-time translation.  
  
    This endpoint checks the waiting list and processes the translation if the model is not busy.  
    """  
    sse_stop_event.clear()  

    async def event_stream():  
        while not sse_stop_event.is_set():  
            if waiting_list and not model.processing:  
                response_data = waiting_list.pop(0)  
                audio_buffer = f"audio/{response_data.times}.wav"  
                o_lang = response_data.ori_lang  
                t_lang = response_data.trans_lang  
  
                try:  
                    # Create an event to signal stopping  
                    stop_event = threading.Event()  
                    # Create timing thread and inference thread  
                    time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, WAITING_TIME))  
                    inference_thread = threading.Thread(target=ws_translate_and_print, args=(model, audio_buffer, o_lang, t_lang, stop_event))  
  
                    # Start the threads  
                    time_thread.start()  
                    inference_thread.start()  
  
                    # Wait for timing thread to complete and stop the inference thread if still running  
                    time_thread.join()  
                    stop_thread(inference_thread)  
  
                    # Remove the audio buffer file  
                    os.remove(audio_buffer)  
  
                    # Process results from the result queue  
                    while not model.result_queue.empty():  
                        o_result, t_result, inference_time, g_translate_time, translate_method = model.result_queue.get()  
                        response_data.ori_text = o_result  
                        response_data.trans_text = t_result  
                        response_data.transcribe_time = inference_time  
                        response_data.translate_time = g_translate_time  
  
                        logger.debug(response_data.model_dump_json())  
                        logger.info(f" | device_id: {response_data.device_id} | audio_uid: {response_data.audio_uid} | language: {response_data.ori_lang} -> {response_data.trans_lang} | translate_method: {translate_method} | ")  
                        logger.info(f" | transcription: {response_data.ori_text} | ")  
                        logger.info(f" | translation: {response_data.trans_text} | ")  
                        logger.info(f" | Inference completed in {inference_time:.2f} seconds. Translation completed in {g_translate_time:.2f} seconds. | ")  
  
                        base_response = BaseResponse(  
                            status="OK",  
                            message=f" | transcription: {response_data.ori_text} | translation: {response_data.trans_text} | ",  
                            data=response_data  
                        )  
                        yield f"{base_response}\n\n"  
                except Exception as e:  
                    logger.error(f' | inference() error: {e} | ')  
                    base_response = BaseResponse(  
                        status="FAILED",  
                        message=f" | inference() error: {e} | ",  
                        data=response_data  
                    )  
                    yield f"{base_response}\n\n"  
            await asyncio.sleep(0.1)  
  
    return StreamingResponse(event_stream(), media_type="text/event-stream") 

@app.post("/stop_sse")  
async def stop_sse():  
    """Endpoint to stop the SSE connection."""  
    sse_stop_event.set() 
    return BaseResponse(status="OK", message=" | SSE connection has been stopped | ", data=None)  

@app.post("/vst_translate", description="**[DEPRECATED]** This endpoint is deprecated and will be removed in the future. Please use `/sse_rtt_translate/v2` instead.")  
async def vst_translate(  
    file: UploadFile = File(...),  
    transcription_request: VSTTranscriptionData = Depends()  
):  
    """  
    Transcribe and translate an audio file.  
      
    This endpoint receives an audio file and its associated metadata, and  
    performs transcription and translation on the audio file.  
      
    :param file: UploadFile  
        The audio file to be transcribed.  
    :param audio_uid: str  
        The unique ID of the audio.  
    :param sample_rate: int  
        The sample rate of audio.  
    :param o_lang: str  
        The original language of the audio.  
    :param t_lang: str  
        The target language for translation.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    # Extract data from transcription_request  
    times = str(datetime.datetime.now())  
    o_lang = transcription_request.o_lang.lower()  
    t_lang = transcription_request.t_lang.lower()  
      
    # Create response data structure  
    response_data = VSTResponseSTT(  
        ori_text="",  
        tar_text="",  
    )  
      
    file_name = times + ".wav"  
    audio_buffer = f"audio/{file_name}"  
    with open(audio_buffer, 'wb') as f:  
        f.write(file.file.read())  
      
    # Check if the audio file exists  
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status="FAILED", message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
      
    # Check if the model has been loaded  
    if model.model_version is None:  
        return BaseResponse(status="FAILED", message=" | model haven't been load successful. may out of memory please check again | ", data=response_data)  
      
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST or t_lang not in LANGUAGE_LIST:  
        logger.info(f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status="FAILED", message=f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
      
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
          
        timeout = transcription_request.timeout  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, timeout))  
        inference_thread = threading.Thread(target=translate_and_print, args=(model, audio_buffer, o_lang, t_lang, result_queue, stop_event))  
          
        # Start the threads  
        time_thread.start()  
        inference_thread.start()  
          
        # Wait for timing thread to complete and check if the inference thread is active to close  
        time_thread.join()  
        stop_thread(inference_thread)  
          
        # Remove the audio buffer file  
        os.remove(audio_buffer)  
          
        # Get the result from the queue  
        if not result_queue.empty():  
            o_result, t_result, inference_time, g_translate_time, translate_method = result_queue.get()  
            response_data.ori_text = o_result  
            response_data.tar_text = t_result  
              
            logger.debug(response_data.model_dump_json())  
            logger.info(f" | language: {o_lang} -> {t_lang} | translate_method: {translate_method} | timeout time: {timeout} | ")  
            logger.info(f" | transcription: {o_result} |")  
            logger.info(f" | translation: {t_result} |")  
            logger.info(f" | inference has been completed in {inference_time:.2f} seconds. | translate has been completed in {g_translate_time:.2f} seconds.")  
            state = "OK"  
        else:  
            logger.info(f" | Inference has exceeded the upper limit time and has been stopped |")  
            state = "FAILED"  
          
        return BaseResponse(status=state, message=f" | transcription: {response_data.ori_text} | translation: {response_data.tar_text} | ", data=response_data)  
    except Exception as e:  
        logger.error(f' | inference() error: {e} | ')  
        return BaseResponse(status="FAILED", message=f" | inference() error: {e} | ", data=response_data)  

@app.post("/vst_translate/v2")  
async def vst_translate(  
    file: UploadFile = File(...),  
    audio_uid: str = Form(...),  
    sample_rate: int = Form(...),  
    o_lang: str = Form(...),  
    t_lang: str = Form(...),  
    timeout: float = Form(...)  
):  
    """  
    Transcribe and translate an audio file.  
      
    This endpoint receives an audio file and its associated metadata, and  
    performs transcription and translation on the audio file.  
      
    :param file: UploadFile  
        The audio file to be transcribed.  
    :param audio_uid: str  
        The unique ID of the audio.  
    :param sample_rate: int  
        The sample rate of the audio.  
    :param o_lang: str  
        The original language of the audio.  
    :param t_lang: str  
        The target language for translation.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    # Extract data from transcription_request  
    times = str(datetime.datetime.now())  
    o_lang = o_lang.lower()  
    t_lang = t_lang.lower()  
      
    # Create response data structure  
    response_data = VSTResponseSTT(  
        ori_text="",  
        tar_text="",  
    )  
      
    file_name = times + ".wav"  
    audio_buffer = f"audio/{file_name}"  
    with open(audio_buffer, 'wb') as f:  
        f.write(file.file.read())  
      
    # Check if the audio file exists  
    if not os.path.exists(audio_buffer):  
        return BaseResponse(status="FAILED", message=" | The audio file does not exist, please check the audio path. | ", data=response_data)  
      
    # Check if the model has been loaded  
    if model.model_version is None:  
        return BaseResponse(status="FAILED", message=" | model haven't been load successful. may out of memory please check again | ", data=response_data)  
      
    # Check if the languages are in the supported language list  
    if o_lang not in LANGUAGE_LIST or t_lang not in LANGUAGE_LIST:  
        logger.info(f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status="FAILED", message=f" | One or both languages are not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
      
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
          
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, timeout))  
        inference_thread = threading.Thread(target=translate_and_print, args=(model, audio_buffer, o_lang, t_lang, result_queue, stop_event))  
          
        # Start the threads  
        time_thread.start()  
        inference_thread.start()  
          
        # Wait for timing thread to complete and check if the inference thread is active to close  
        time_thread.join()  
        stop_thread(inference_thread)  
          
        # Remove the audio buffer file  
        os.remove(audio_buffer)  
          
        # Get the result from the queue  
        if not result_queue.empty():  
            o_result, t_result, inference_time, g_translate_time, translate_method = result_queue.get()  
            response_data.ori_text = o_result  
            response_data.tar_text = t_result  
              
            logger.debug(response_data.model_dump_json())  
            logger.info(f" | language: {o_lang} -> {t_lang} | translate_method: {translate_method} | timeout time: {timeout} | ")  
            logger.info(f" | transcription: {o_result} |")  
            logger.info(f" | translation: {t_result} |")  
            logger.info(f" | inference has been completed in {inference_time:.2f} seconds. | translate has been completed in {g_translate_time:.2f} seconds.")  
            state = "OK"  
        else:  
            logger.info(f" | Inference has exceeded the upper limit time and has been stopped |")  
            state = "FAILED"  
          
        return BaseResponse(status=state, message=f" | transcription: {response_data.ori_text} | translation: {response_data.tar_text} | ", data=response_data)  
    except Exception as e:  
        logger.error(f' | inference() error: {e} | ')  
        return BaseResponse(status="FAILED", message=f" | inference() error: {e} | ", data=response_data)  

@app.post("/text_translate")  
async def text_translate(  
    translate_request: TextData = Form()  
):  
    """  
    Translate a text.  
  
    This endpoint receives text and its associated metadata, and performs translation on the text.  
  
    :param translate_request: TextData  
        The request containing the text to be translated.  
    :rtype: BaseResponse  
        A response containing the translation results.  
    """  
    o_lang = translate_request.o_lang.lower()  
    t_lang = translate_request.t_lang.lower()  
    o_result = translate_request.ori_text  
  
    # Create response data structure  
    response_data = VSTResponseSTT(  
        ori_text="",  
        tar_text="",  
    )  
  
    try:  
        # Perform translation  
        translated_pred, g_translate_time, translate_method = model.translate(o_result, o_lang, t_lang)  
        response_data.ori_text = o_result  
        response_data.tar_text = translated_pred  
          
        # Log the translation details  
        logger.info(f" | language: {o_lang} -> {t_lang} | translate_method: {translate_method} | translate has been completed in {g_translate_time:.2f} seconds. |")  
        logger.info(f" | transcription: {o_result} |")  
        logger.info(f" | translation: {translated_pred} |")  
          
        state = "OK"  
        return BaseResponse(status=state, message=f" | input text: {o_result} | translation: {translated_pred} | ", data=response_data)  
    except Exception as e:  
        logger.error(f' | inference() error: {e} | ')  
        translated_pred = o_result  
        state = "FAILED"  
        return BaseResponse(status=state, message=f" | inference() error: {e} | ", data=response_data)  

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
                logger.info(f" | Deleted old file: {file_path} | ")  
  
# Daily task scheduling  
def schedule_daily_task(stop_event):  
    while not stop_event.is_set():  
        if local_now.hour == 0 and local_now.minute == 0:  
            delete_old_audio_files()  
            time.sleep(60)  # Prevent triggering multiple times within the same minute  
        time.sleep(1)  
  
# Start daily task scheduling  
service_stop_event = Event()  
task_thread = Thread(target=schedule_daily_task, args=(service_stop_event,))  
task_thread.start()  
  
@app.on_event("shutdown")  
def shutdown_event():  
    service_stop_event.set()  
    task_thread.join()  
    model.ollama_translator.close()
    logger.info(" | Scheduled task has been stopped. | ")  
  
if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"  
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)   
    
    
 

