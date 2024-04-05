from zalo_tts import ZaloTTS
from gtts import gTTS
import re

text = "hello word!"
letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                          " ",          # Replace all non-letters with spaces
                          str(text))

# curl -H "apikey: your_api_key_here" -X POST "https://api.zalo.ai/{{target_api_url}}"
def ZaloMouth (text):
    tts = ZaloTTS(speaker=ZaloTTS.NORTHERN_WOMEN, api_key="KTM30eBmOnF7zVxvfwURXA3KsYvXK6l7")

    tts.text_to_speech(text)






# demo(text)