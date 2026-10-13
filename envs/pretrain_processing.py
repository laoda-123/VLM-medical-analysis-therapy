from keybert import KeyBERT
import numpy as np
import re 
from env_builder import load_env_file
import jieba
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from keybert import KeyBERT
from flask import app,request,render_template,redirect,Response,Flask,jsonify

app=Flask(__name__)

with open("H:/pretained/stopwords.txt","r+") as f :
   stop_words=f.readlines()
data ={}

env_config=load_env_file("J:/image_language_model/keybert.yaml")
kw_model=env_config["keyBert"][1]
def tokenize_text(text):
   total_token={}
   eng_word=re.findall('[a-zA-Z]+',text,re.M)
   eng_tokens=word_tokenize(eng_word)
   ch_tokens=re.findall(r'[\u4e00-\u9fff]',text)
   
   tokens=jieba.lcut(text)
   tokens=[token.strip('/') for token in tokens]
   return tokens,eng_tokens 
def remove_special_characters(text):
   eng_tokens,ch_tokens=tokenize_text(text)
   pattern=r"[\[\]\{\}\<\>\"\\\n\r"
   ch_clean_tokens=re.sub(pattern,'',ch_tokens)
   eng_clean_tokens=re.sub(pattern,'',eng_tokens)
   clean_tokens=ch_clean_tokens+eng_clean_tokens
   return clean_tokens  
def  remove_stopwords(text):
   str=''
   tokens=tokenize_text(text)
   stop_words_eng = set(stopwords.words('english'))
   stop_words_ch=set(stopwords.words('chinese'))
   stop_words=stop_words_eng.union(stop_words_ch)
   filter_tokens =[token for token in tokens if token not in stop_words]
   for  filter_token in filter_tokens:
      str=str.join(filter_token)
   return str
def normalize_corpus(file_path,output_path,tokenize=False):
   normalize_corpus=[] 
   with open(file_path,"a+") as corpus:
    for text in corpus.readlines():
       text=remove_special_characters(text)
       text=remove_stopwords(text)
       normalize_corpus.append(text)
       if tokenize:
           text= tokenize_text(text)
           normalize_corpus.append(text)
    corpus.write(output_path)
   return normalize_corpus

#user input their text
@app.route('/text_input',methods=["POST"])
def get_word():
   text=request.form.get('user_input')
   tokenize=request.form.get('tokenize')
   normalize_corpus=[]
   text=remove_special_characters(text)
   text=remove_stopwords(text)
   if tokenize=="Yes":
        text=tokenize_text(text)
        normalize_corpus.extend(text)
   return normalize_corpus


#train_input
@app.route('/insert_text',methods=["POST"])
def get_text_image_input():
   train_text=request.form.get('train_text_input','')
   files=request.files.getlist('fileInput')
   tokenize=request.form.get('tokenize-text')
   normalize_corpus=[]
   for line in train_text:
      text=remove_special_characters(line)
      text=remove_stopwords(text)
      normalize_corpus.extend(text)
      if tokenize=="Yes":
         keywords = kw_model.extract_keywords(
         normalize_corpus)
        
      else:
         keywords=normalize_corpus
   return  {"result":keywords[:5]}