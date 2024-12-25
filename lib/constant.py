from pydantic import BaseModel
import torch
from datetime import datetime

#############################################################################

class ModlePath(BaseModel):
    large_v2: str = "models/large-v2.pt"
    medium: str = "/mnt/models/medium.pt"
    # turbo: str = "models/large-v3-turbo.pt"

#############################################################################
""" options for Whisper inference """
OPTIONS = {
    "fp16": torch.cuda.is_available(),
    "language": "en",
    "task": "transcribe",
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.2,
}

#############################################################################

class TranscriptionData(BaseModel):
    meeting_id: str
    device_id: str
    audio_uid: str
    times: datetime
    o_lang: str
    t_lang: str

#############################################################################

class ResponseSTT(BaseModel):
    meeting_id: str
    device_id: str
    ori_lang: str
    ori_text: str
    trans_lang: str
    trans_text: str
    times: datetime
    audio_uid: str
    transcribe_time: float
    translate_time: float

#############################################################################

LANGUAGE_LIST = ['zh', 'en', 'ja', 'ko', "de", "es"]

#############################################################################

# google or argos or gpt-4o
TRANSLATE_METHODS = ['google', 'argos', 'gpt-4o']

#############################################################################

AZURE_CONFIG = '/mnt/lib/azure_config.yaml'

#############################################################################

# GPT-4o prompt
# LANGUAGE_QUEUE = ["zh", "en", "ja", "ko", "de", "es"]

SOURCE_LANGUAGE = [  
    ["繁體中文", "Traditional Chinese", "繁体字中国語", "중국어 번체", "Traditionelles Chinesisch", "Chino Tradicional"],  
    ["英文", "English", "英語", "영어", "Englisch", "Inglés"],  
    ["日文", "Japanese", "日本語", "일본어", "Japanisch", "Japonés"],  
    ["韓文", "Korean", "韓国語", "한국어", "Koreanisch", "Coreano"],  
    ["德文", "German", "ドイツ語", "독일어", "Deutsch", "Alemán"],  
    ["西班牙文", "Spanish", "スペイン語", "스페인어", "Spanisch", "Español"]  
]  

SYSTEM_PRMOPT = {"zh": "請將以下[source_language]翻譯成[繁體中文]，並確保翻譯文本的語句流暢性與文法格式正確。翻譯風格應該適合一般民眾閱讀，保持自然且易於理解。 如果有任何專有名詞或文化特定的內容，請進行適當的本地化處理。",
                 "en": "Please translate the following [source_language] into [English], ensuring that the translated text is fluent and grammatically correct. The translation style should be suitable for the general public, maintaining a natural and easy-to-understand tone. If there are any technical terms or culturally specific content, please localize them appropriately.",
                 "ja": "以下の[source_language]を[日本語]に翻訳してください。翻訳されたテキストの文法と文の流れが正しいことを確認してください。翻訳スタイルは一般の人々が読むのに適したもので、自然で理解しやすいものにしてください。専門用語や文化特有の内容がある場合は、適切にローカライズしてください。",
                 "ko": "다음 [source_language]를 [한국어]로 번역해 주세요. 번역된 텍스트는 유창하고 문법적으로 정확해야 합니다. 번역 스타일은 일반 대중에게 적합하도록 자연스럽고 이해하기 쉬운 어조를 유지해야 합니다. 기술 용어 또는 문화적으로 특정한 내용이 있는 경우 적절하게 현지화해 주세요.",
                 "de": "Bitte übersetzen Sie den folgenden [source_language] ins [Deutschen] und achten Sie darauf, dass der übersetzte Text flüssig und grammatikalisch korrekt ist. Der Stil der Übersetzung sollte für die breite Öffentlichkeit geeignet sein und einen natürlichen und leicht verständlichen Tonfall beibehalten. Falls es Fachbegriffe oder kulturspezifische Inhalte gibt, lokalisieren Sie diese bitte entsprechend.",
                 "es": "Por favor, traduzca el siguiente [source_language] a [español] y asegúrese de que el texto traducido es fluido y gramaticalmente correcto. El estilo de la traducción debe ser adecuado para el público en general y mantener un tono natural y fácilmente com",}

#############################################################################
