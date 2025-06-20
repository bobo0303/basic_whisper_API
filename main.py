from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Form, Depends
from fastapi.responses import StreamingResponse  
import os  
import time  
import pytz  
import logging  
import uvicorn  
import datetime  
import threading 
from queue import Queue  
from threading import Thread, Event  
from api.model import Model  
from api.threading_api import transcribe_and_print, waiting_times, stop_thread
from lib.base_object import BaseResponse  
from lib.constant import ResponseSTT, LoadModelRequest, LANGUAGE_LIST, ASR_METHODS
  
#############################################################################  

os.environ['TZ'] = 'Asia/Taipei'  
time.tzset()
  
if not os.path.exists("./audio"):  
    os.mkdir("./audio")  
if not os.path.exists("./logs"):  
    os.mkdir("./logs")  
    
# Configure logging  
log_format = "%(asctime)s - %(message)s"  
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

##############################################################################  
  
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
    logger.info(f" | ############################################################### | ")  
    return BaseResponse(message=f" | inference model: You can choose {ASR_METHODS} | ", data=None)  


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
  

    
@app.post("/transcribe")  
async def transcribe(  
    file: UploadFile = File(...),  
    patience: int = Form(3),  
    language: str = Form("ZH"),  
):  
    """  
    Transcribe an audio file.  
      
    This endpoint receives an audio file and its associated metadata, and  
    performs transcription on the audio file.  
      
    :param file: UploadFile  
        The audio file to be transcribed.  
    :param patience: int  
        The transcribe uplimit time.  
    :rtype: BaseResponse  
        A response containing the transcription results.  
    """  
    language = language.lower()  # Ensure the language is in lowercase

    # Create response data structure  
    response_data = ResponseSTT(  
        language=language,  
        text="",  
        transcribe_time=0.0,  
    )  
  
    # Save the uploaded audio file  
    times = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
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
    if language not in LANGUAGE_LIST:  
        logger.info(f" | The language is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ")  
        return BaseResponse(status="FAILED", message=f" | The language {language} is not in LANGUAGE_LIST: {LANGUAGE_LIST}. | ", data=response_data)  
  
    try:  
        # Create a queue to hold the return value  
        result_queue = Queue()  
        # Create an event to signal stopping  
        stop_event = threading.Event()  
  
        # Create timing thread and inference thread  
        time_thread = threading.Thread(target=waiting_times, args=(stop_event, model, patience))  
        inference_thread = threading.Thread(target=transcribe_and_print, args=(model, audio_buffer, result_queue, language, stop_event))  
  
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
            o_result, inference_time = result_queue.get()  
            response_data.text = o_result  
            response_data.transcribe_time = inference_time  
            logger.debug(response_data.model_dump_json())  
            logger.info(f" | transcription: {response_data.text} |")  
            logger.info(f" | inference has been completed in {inference_time:.2f} seconds. | language: {language} | ")  
            state = "OK"  
        else:  
            logger.info(f" | Inference has exceeded the upper limit time and has been stopped |")  
            state = "FAILED"  
  
        return BaseResponse(status=state, message=f" | transcription: {response_data.text} | ", data=response_data)  
    except Exception as e:  
        logger.error(f" | inference() error: {e} | ")  
        return BaseResponse(status="FAILED", message=f" | inference() error: {e} | ", data=response_data)  
    

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
    logger.info(" | Scheduled task has been stopped. | ")  
  
if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"  
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)   
    
    
 

