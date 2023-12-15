from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason

import requests

#TTS/ STT
def speech_to_text(api_key, resource_name, deployment_id, api_version, audio_file_path):
    
    url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/audio/transcriptions?api-version={api_version}"

    headers = {
        "api-key": api_key,
    }

    files = {
        "file": open(audio_file_path, "rb"),
        "language": (None, "zh"),
        "prompt": (None, "you are a voice message assistant excel at answering guests questions, your task is focus on sentences fluency and correct the typos."),
        "temperature": (None, "0.2"),
        "response_format": (None, "json")
    }

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.text  # 或 response.json()，根據響應格式
    else:
        return f"Error: {response.status_code}, {response.text}"




def text_to_speech(api_key, region, text):
    print('移除網址後的文字:',text)
    #創建一個語音配置對象
    speech_config = SpeechConfig(subscription=api_key, region=region)
    #設置語言為中文
    speech_config.speech_synthesis_language = "zh-CN"  
    speech_config.speech_synthesis_voice_name = "zh-TW-YunJheNeural"
    #創建一個語音合成器對象
    synthesizer = SpeechSynthesizer(speech_config=speech_config)
    #調用語音合成器，speak_text_async 方法將文本轉換為語音。
    result = synthesizer.speak_text_async(text).get()

    if result.reason == ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    else:
        return f"Error: {result.reason}"