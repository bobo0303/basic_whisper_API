from pydantic import BaseModel
import torch

#############################################################################

class ModlePath(BaseModel):
    large_v2: str = "./models/large-v2.pt"
    medium: str = "./models/medium.pt"
    sensevoice: str = "./models/SenseVoiceSmall"
    punc: str = "./models/ct-punc"

#############################################################################
""" options for Whisper inference """
OPTIONS = {
    "fp16": torch.cuda.is_available(),
    "language": None,
    "task": "transcribe",
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6, # default 0.6 | ours 0.2
}

SV_OPTIONS = {
    "language": "auto",
    "itn": True,
    "ban_emo_unk": False,
}

SENSEVOCIE_PARMATER = {"model": "./models/SenseVoiceSmall",
                        "disable_update": True,
                        "disable_pbar": True,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",            
                        }

PUNC_PARMATER = {"model": "./models/ct-punc",
                        "disable_update": True,
                        "disable_pbar": True,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",            
                        }

IS_PUNC = True

#############################################################################

# Request body model for loading a model
class LoadModelRequest(BaseModel):
    models_name: str
    
# Request for loading new translate method
class LoadMethodRequest(BaseModel):
    method_name: str

#############################################################################

class ResponseSTT(BaseModel):
    language: str
    text: str  
    transcribe_time: float
    
#############################################################################

LANGUAGE_LIST = ['zh', 'en', 'ja', 'ko', "de", "es"]

#############################################################################

ASR_METHODS = ['medium', 'large_v2', 'sensevoice']

#############################################################################
