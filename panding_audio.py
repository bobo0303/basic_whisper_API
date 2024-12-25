from pydub import AudioSegment  
  
# 加载2秒的音频文件  
input_audio = AudioSegment.from_file("id_1_2024-09-12_14_50_12.499973.wav")  
  
# 计算需要填充的时长  
target_duration_ms = 30 * 1000  # 30秒转换为毫秒  
input_duration_ms = len(input_audio)  
padding_duration_ms = target_duration_ms - input_duration_ms  
  
if padding_duration_ms > 0:  
    # 生成填充的空白音频  
    padding_audio = AudioSegment.silent(duration=padding_duration_ms)  
  
    # 将空白音频添加到原始音频的末尾  
    output_audio = input_audio + padding_audio  
  
    # 保存新的音频文件  
    output_audio.export("output_audio4.wav", format="wav")  
else:  
    print("输入音频已超过30秒，无需填充。") 