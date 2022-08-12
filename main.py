from email.mime import audio
from msilib.schema import Directory
from urllib import request
from fastapi import FastAPI, Body, Request, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import nltk
import librosa
import torch
import gradio as gr
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

###########################################################################################################
#Audio Transcript code

model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def load_data(speech, sample_rate):

  #reading the file
  #speech, sample_rate = librosa.load(input_file)
  #make it 1-D
  if len(speech.shape) > 1: 
      speech = speech[:,0] + speech[:,1]
  #Resampling the audio at 16KHz
  if sample_rate !=16000:
    speech = librosa.resample(speech, sample_rate,16000)
  return speech

def correct_casing(input_sentence):

  sentences = nltk.sent_tokenize(input_sentence)
  return (' '.join([s.replace(s[0],s[0].capitalize(),1) for s in sentences]))

def asr_transcript(speech, sample_rate):

  speech2 = load_data(speech, sample_rate)
  #Tokenize
  input_values = tokenizer(speech2, return_tensors="pt").input_values
  #Take logits
  logits = model(input_values).logits
  #Take argmax
  predicted_ids = torch.argmax(logits, dim=-1)
  #Get the words from predicted word ids
  transcription = tokenizer.decode(predicted_ids[0])
  #Correcting the letter casing
  transcription = correct_casing(transcription.lower())
  return transcription

##############################################################################################################

app = FastAPI()
templates= Jinja2Templates(directory="html")

@app.get("/home")
def audiotrans(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/submitaudio")
async def handle_form(audiofile: UploadFile = File(...)):
    speech, sample_rate = librosa.load(audiofile.file)
    print(audiofile.filename)
    speak = asr_transcript(speech, sample_rate)
    print(speak)
    return {"transcript":speak}


