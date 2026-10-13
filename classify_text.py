import  numpy as  np 
import torch.nn   as nn 
import os 
from keybert import KeyBERT
from PIL import Image
from envs.env_builder import load_env_file
import re
from VitExtractor import extract_keyword

from envs.env_builder import load_env_file
import torch
from video_recognition import encoder
dict=load_env_file("J:/image_language_model/config/config.yaml")
CT_scanning={"pulmonary_ill":["Pulmonary_emphysema", "Pneumonia", "Pulmonary_nodule", "Lung_cancer", "Tuberculosis_of_the_lung", "Pulmonary_alveolar_hyperplasia"],"disease_of_pancreas":["Pancreatitis", "Pancreatic_tumor", "Pancreatic_cyst"]}




def resize(image_root,height,width):
    for filename in os.listdir(image_root):
     if filename.endswith(('.png','.jpg','.gif')):
        img=Image.open(os.path.join(image_root,filename))
        resized=img.resize((height,width))
        resized.save(os.path.join(image_root,filename))
def classify(text_root,image_root,npz_path): 
    target=[]
    scissors_key=[]
    tweezers_key=[]
    tray_key=[]
    cotton_ball_key=[]
    shallow_key=[]
    injection_key=[]
    iodine_key=[]
    drip_key=[]
    list=[] 
    for file in os.listdir(text_root):
      if file.endswith((".txt")):
       name,_=os.path.splitext(file)
       cl=re.sub(r"\d+",'',name)
       if cl=="scissors":
        with open (os.path.join(text_root,file),"r+",encoding="utf-8") as f :
         lines1=f.read()
         lines1.strip('\n')
         lines1.strip('')
         scissors_key=extract_keyword(text=lines1,top_n=5)
         f.close()
       if cl=="tweezers":
        with open(os.path.join(text_root,file),"r+",encoding="utf-8") as f:
          lines2=f.read()
          lines2.strip('\n')
          lines2.strip('')
          tweezers_key=extract_keyword(text=lines2,top_n=5)
          f.close()
       if cl=="shallow":
        with open(os.path.join(text_root,file),"r+",encoding="utf-8") as f:
          lines3=f.readline()
          lines3.strip('\n')
          lines3.strip('')
          shallow_key=extract_keyword(text=lines3,top_n=5)
          f.close()
       if cl=="tray":
        with open(os.path.join(text_root,file),"r+",encoding="utf-8") as f:
         lines4=f.read()
         lines4.strip('\n')
         lines4.strip('')
         tray_key=extract_keyword(text=lines4,top_n=5)
         f.close()
       if cl=="cotton_ball":
        with open(os.path.join(text_root,file),"r+",encoding="utf-8") as f:
          lines5=f.read()
          lines5.strip('')
          lines5.strip('\n')
          cotton_ball_key=extract_keyword(text=lines5,top_n=5)
          f.close()
       if cl=="injection_syringe":
        with open(os.path.join(text_root,file),"r+",encoding="utf-8") as f:
          lines6=f.read()
          lines6.strip('\n')
          lines6.strip('')
          injection_key=extract_keyword(text=lines6,top_n=5)
          f.close()
       if cl=="iodine":
        with open(os.path.join(text_root,file),"r+",encoding="utf-8") as f:
          lines7=f.read()
          lines7.strip('\n')
          lines7.strip('')
          iodine_key=extract_keyword(text=lines7,top_n=5)
          f.close()
       if cl=="drip":
        with open(os.path.join(text_root,file),"r+",encoding="utf-8") as f:
          lines8=f.read()
          lines8.strip('')
          lines8.strip('\n')
          drip_key=extract_keyword(text=lines8,top_n=5)
          f.close()
   
    
    
    for image  in os.listdir(image_root):
       if  image.endswith(('.png','.jpg','.jpeg','.gif')):
           name,_=os.path.splitext(image)
           class_in=re.sub(r"\d+",'',name)
          
           if class_in in dict["cotton_ball"]:
             item_cotton_ball=[]
             target.append(os.path.join(image_root,image))
             for element in cotton_ball_key:
               item_cotton_ball.append(element[0])
             list.append(encoder(item_cotton_ball))
           if class_in in dict["injection_syringe"]:
             item_injection_syringe=[]
             
             target.append(os.path.join(image_root,image))
             for element in injection_key:
              item_injection_syringe.append(element[0])
             list.append(encoder(item_injection_syringe))
           if class_in in dict["iodine"]:
             item_iodine=[]
           
             target.append(os.path.join(image_root,image))
             for element in iodine_key:
                item_iodine.append(element[0])
             list.append(encoder(item_iodine))

           if class_in  in dict["scissors"]:
             item_scissors=[]
           
             target.append(os.path.join(image_root,image))
             for element in scissors_key:
               item_scissors.append(element[0])
             
             list.append(encoder(item_scissors))
           if class_in  in dict["tweezers"]:
             item_tweezers=[]
           
             target.append(os.path.join(image_root,image))
             for element in tweezers_key:
               item_tweezers.append(element[0])
             list.append(encoder(item_tweezers))
           '''
           if class_in in dict["shallow"]:
             transform = transforms.Compose([
                     transforms.ToTensor(),  # 自动将uint8转为float32，像素值归一化到[0,1]
                   ])
          
             img=Image.open(os.path.join(image_root,image)).convert("RGBA")
             img_arr=np.array(img)
             img_tensor=transform(img_arr)
             list.append(shallow_key[0])
           '''
           if class_in in dict["tray"]:
           
             item_tray=[]
             for element in tray_key:
                 item_tray.append(element[0])
             target.append(os.path.join(image_root,image))
             list.append(encoder(item_tray))
           if class_in  in  dict["drip"]:
             item_drip=[]
             for element in drip_key:
               item_drip.append(element[0])
             list.append(encoder(item_drip))
             target.append(os.path.join(image_root,image))
  
    input=torch.from_numpy(np.array(list))
    np.savez_compressed(npz_path,test=input,figure=target,allow_pickle=True)
    
def main():
    text_root="J:/scalpel"
    image_root="K:/scalpel"
    classify(text_root,image_root,"I:/diffuser_model/data.npz")

if __name__=='__main__':
   main()