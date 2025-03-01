from gtts import gTTS

def text_to_mp3(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    print(f"MP3 file '{filename}' has been created successfully.")

if __name__ == "__main__":
    text_input = "Hello, welcome to my Python application!"
    output_file = "output.mp3"
    
    text_to_mp3(text_input, output_file)
