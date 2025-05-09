import re
import os
import time
import sys
import string

  
def extract_sensevoice_result_text(input_string: str) -> str:  
    """  
    Extract the text after the last '>' character in the input string.  
  
    Args:  
        input_string (str): The input string to process.  
  
    Returns:  
        str: The extracted text or an empty string if no match is found.  
    """  
    pattern = r'>([^<]+)$'  
    match = re.search(pattern, input_string)
    if match:  
        return match.group(1)  
    return ""
  