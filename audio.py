from subprocess import call
import pyaudio
from os import system
from time import sleep
import speech_recognition as sr



r = sr.Recognizer()

index = pyaudio.PyAudio().get_device_count() - 1
print(index)

counter = 0
#print("say something2")
while True:
     with sr.Microphone() as source:                # use the default microphone as the audio source
          print("Adjusting for background noise. One second")
          r.adjust_for_ambient_noise(source)
          print("Say something!")
       	  audio = r.listen(source,timeout = 3)
       	  print('Recognising audio...')
          words = r.recognize_google(audio, language='en-IN', show_all=True)
          print(words)
          if not words:
              continue
          if 'on' in words['alternative'][0]['transcript'] and counter == 0:
          #if True:
              system('gnome-terminal --title="demo" -x python3 ~/PycharmProjects/yogamirror/demo.py')
              counter += 1
          if 'off' in words['alternative'][0]['transcript'] and counter != 0:
              call(['killall', 'demo'])
          if 'pause' in words['alternative'][0]['transcript'] and counter != 0:
              