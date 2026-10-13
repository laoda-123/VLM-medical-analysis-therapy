import sqlite3
import torch.nn as nn 
import torch
import cv2

from  PIL import Image
from envs.env_builder import load_env_file
import numpy as np
import os
import re
from keybert import KeyBERT
from VitExtractor import extract_keyword




bert_config=load_env_file("J:/image_language_model/keybert.yaml")
kw_model=bert_config["keyBert"][1]

def histogram_distance(frame): 
   hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV_FULL)
   hist=cv2.calcHist([hsv],[0,1],None,[50,60],[0,180,0,256]) #std/mean
   return hist


   
def adaptive_threshold_detection(corpus_path,video_path, window_size=10):
    cap=cv2.VideoCapture(video_path)
    prev_hist=None
    boundaries=[]
    frame_idx=0 # frame startage from zero
    while True:
       ret,frame=cap.read()
       if not ret:  #ret=False 表示的是读取结束或者是读取错误
          break
       cv2.imshow('Video',frame
              )
       curr_hist=histogram_distance(frame)
       if prev_hist  is not None:
          similarity=cv2.compareHist(prev_hist,curr_hist,cv2.HISTCMP_CORREL)
       if similarity > 0.95:
           boundaries.append((frame_idx,similarity))
       prev_hist=curr_hist
       frame_idx+=1
    cap.release()
    return boundaries
def extract_frame(video_path,interval,text_path,npz_dir,output_dir):
   
   key_list=[]
   cap=cv2.VideoCapture(video_path)
   if not cap.isOpened():
      print(f"sorry, we can't open  your video")
      return None
   fps=cap.get(cv2.CAP_PROP_FPS)
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   frame_idx = 0
   frame_total=[]
   with open(text_path,mode="a+",encoding="utf-8") as txt: 
      lines=txt.read()
      re.sub(r'\d+\?\!\~\<\>\{\}\[\]\\\n','',lines)
      lines.replace('\n','')
   key=extract_keyword(lines,top_n=10)
   for item in key:
       key_list.append(item[0])
   saved_idx=0
   while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx%interval==0:
            
   # 保存帧（PNG格式保持质量）
         frame_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.png")
         cv2.imwrite(frame_path, frame)
         frame_arr=np.array(frame)
         frame_total.append(frame_arr)
         saved_idx+=1
        frame_idx+= 1
   np.savez_compressed(npz_dir,input_text=key_list,output_frame=frame_total)
   cap.release()
   print(np.array(frame_total).shape)
   return frame_total

   
def main():
  
   extract_frame("K:/爱拍(360联运)/Datas/save/video/20260514-2145.mp4",2,"J:/video/基本特征.txt","J:/video/humanoid_robot.npz","J:/video/humanoid_robot")
if  __name__=='__main__':
   main()