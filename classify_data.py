import torch.nn as nn
import torch 
import cv2
import numpy as np
from PIL import Image 
import os
import re
import matplotlib.pyplot as plt

from datetime import datetime
from envs.env_builder import load_env_file
from transformers import CLIPModel,CLIPProcessor

from sentence_transformers import SentenceTransformer
from torchvision import transforms
from envs.env_run  import run_encoder_decoder
from envs.env_run import run_clip_transformer



#text_encoder
def watershed_contour_extract(image_path,output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 开运算去除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 距离变换
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # 查找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary, sure_fg)
    
    # 标记区域
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    # 分水岭变换
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]  # 标记边界
    
    # 提取轮廓
    contours, _ = cv2.findContours(np.uint8(markers == -1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
      max_contours=max(contours,key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(max_contours)
        
        # 提取轮廓内的图像
      contour_img = img[y:y+h, x:x+w]
        
        # 保存提取的图像
      cv2.imwrite(output_path, contour_img)
      print(f"轮廓内图像已保存到: {output_path}")
      
    else:
      print("未找到任何轮廓")
      

def batch_rgba_convert_rgb(input_path,output_path,bg_color=(200,200,200)):
          _,ext=os.path.splitext(input_path)
          if ext.lower() in [".gif",".jpg",".png"]:
            with Image.open(input_path) as img:
              
               if img.mode == 'RGBA':
                        # 将RGBA转为RGB
                  rgb_img = Image.new("RGB",img.size,bg_color)
                  rgb_img.paste(img, mask=img.split()[3])
                  rgb_img.save(output_path, quality=95)
                  print(f"转换完成: {_}")
                  rgb_img.resize((224,224))
                  return rgb_img
               else:
                  img1=img.resize((224,224),Image.Resampling.LANCZOS)
                  img1.convert("RGB")      
                  img1.save(output_path, quality=95)
                  return img1
def  classify(text_root,image_root,output_folder,bg_color,config1_path,config2_path,img_sequence,sequence): # config1 is image_description.yaml config2  is config.yaml
       processor,model,tokenizer=run_clip_transformer(config1_path,img_sequence)
       encoder,tokenizer1=run_encoder_decoder(config1_path,sequence)
       config1=load_env_file(config1_path)
       dict1=load_env_file(config2_path)
       list1=[]
       list2=[]
       list3=[]
       img_features=[]
       len_list=[]
       len1_list=[]
       for filename in os.listdir(image_root):
        if filename.lower().endswith((".gif",".png",".jpg")):
              batch_rgba_convert_rgb(os.path.join(output_folder,filename),os.path.join(image_root,filename),bg_color=bg_color)
       
       for filename in os.listdir(image_root):
         if filename.lower().endswith((".gif",".png",".jpg")):
           name,_=os.path.splitext(filename)
           hanzi=re.sub(r'\d+', '',name)
           hanzi=hanzi.replace(" ","")
           text=""
           list3.append(os.path.join(image_root,filename))
           if hanzi in dict1["scissors"]:      
              txt=open(os.path.join(text_root,"scissors.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"scissors_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs =processor(images=img)
               with  torch.no_grad():
                 features=model.get_image_features(**inputs)
                 
                
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)
               
                    
         #  text["剪刀"]=txt.readline()
           if hanzi in dict1["tweezers"]:  
            
              txt=open(os.path.join(text_root,"tweezers.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"tweezers_indication.txt"),mode="r+",encoding='utf-8')
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs =processor(images=img)
               
               with  torch.no_grad():
                 image_features=model.get_image_features(**inputs)
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)
              
                  
           if hanzi in dict1["tray"]:
              txt=open(os.path.join(text_root,"tray.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"tray_indication.txt"),mode="r+",encoding='utf-8')
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs =processor(images=img)
               with  torch.no_grad():
                 image_features =model.get_image_features(**inputs )
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)
        
           if hanzi in dict1["drip"]:
              txt=open(os.path.join(text_root,"drip.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"drip_indication.txt"),mode="r+",encoding="urf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs =processor(images=img)
               with  torch.no_grad():
                 image_features =model.get_image_features(**inputs)
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)
                  
           if hanzi in dict1["iodine"]:
              txt=open(os.path.join(text_root,"iodine.txt"),mode="r+",encoding="utf-8")
              indication=open(os.path.join(text_root,"iodine_indication.txt"),mode="r+",encoding='utf-8')
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs=processor(img)
               with  torch.no_grad():
                 image_features =model.get_image_features(**inputs)
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)

           if hanzi in dict1["injection_syringe"]:
              txt=open(os.path.join(text_root,"injection_syringe.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"injection_syringe_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs= processor(img)
               with  torch.no_grad():
                 
         
                 image_features =model.get_image_features(**inputs)
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)
  

         
           if hanzi in dict1["CT_scanning"]["pulmonary_ill"] or hanzi in dict1["CT_scanning"]["disease_of_pancreas"]:
            if hanzi=="Pulmonary_emphysema":
              txt=open(os.path.join(text_root,"Pulmonary_emphysema.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Pulmonary_emphysema_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs=processor(img)
               with  torch.no_grad():
                 image_features=model.get_image_features(**inputs)
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)
                    
            if hanzi=="Pneumonia":
              txt=open(os.path.join(text_root,"Pneumonia.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Pneumonia_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs=processor(img)
               with  torch.no_grad():
                 image_features=model.get_image_features(**inputs)
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)

               
                  
            if hanzi=="Pulmonary_nodule":
              txt=open(os.path.join(text_root,"Pulmonary_nodule.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Pulmonary_nodule_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs =processor(images=img)
               with  torch.no_grad():
                 
                 image_features =model.get_image_features(**inputs)
                 img_features.append(image_features)
               list1.append(text)
               list2.append(indication)
            if hanzi=="Lung_cancer":
              txt=open(os.path.join(text_root,"Lung_cancer.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Lung_cancer_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs =processor(images=img)
               with  torch.no_grad():
                                
                    image_features =model.get_image_features(**inputs)
                    img_features.append(image_features)
               list1.append(text)
               list2.append(indication)

            if hanzi=="Tuberculosis_of_the_lung":
              txt=open(os.path.join(text_root,"Tuberculosis_of_the_lung.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Tuberculosis_of_the_lung_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
               img=Image.open(os.path.join(image_root,filename)).convert("RGB")
               inputs =processor(images=img)
               with  torch.no_grad():                      
                    image_features =model.get_image_features(**inputs)
                    img_features.append(image_features)
                
               list1.append(text)
               list2.append(indication)
            if hanzi=="Pulmonary_alveolar_hyperplasi":
              txt=open(os.path.join(text_root,"Pulmonary_alveolar_hyperplasia.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Pulmonary_alveolar_hyperplasia_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
                img=Image.open(os.path.join(image_root,filename)).convert("RGB")
                inputs =processor(images=img)
                with  torch.no_grad():                      
                      image_features =model.get_image_features(**inputs)
                      img_features.append(image_features)
                list1.append(text)
                list2.append(indication)
          
            if  hanzi=="Pancreatitis":
              txt=open(os.path.join(text_root,"Pancreatitis.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Pancreatitis_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
                img=Image.open(os.path.join(image_root,filename)).convert("RGB")
                inputs =processor(images=img)
                with  torch.no_grad():                      
                       image_features =model.get_image_features(**inputs)
                       img_features.append(image_features)
                list1.append(text)
                list2.append(indication)
                    
            if hanzi =="Pancreatic_tumor":
              txt=open(os.path.join(text_root,"Pancreatic_tumor.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Pancreatic_tumor_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
                img=Image.open(os.path.join(image_root,filename)).convert("RGB")
                inputs =processor(images=img)
                with  torch.no_grad():                      
                    image_features =model.get_image_features(**inputs)
                    img_features.append(image_features)
           # 获取图像Token的隐藏状态（特征向量）
                list1.append(text)
                list2.append(indication)
            if hanzi =="Pancreatic_cyst":
              txt=open(os.path.join(text_root,"Pancreatic_cyst.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"Pancreatic_cyst_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
                img=Image.open(os.path.join(image_root,filename)).convert("RGB")
      
                inputs =processor(images=img)
                with  torch.no_grad():                      
                    image_features =model.get_image_features(**inputs)
                    img_features.append(image_features)
                list1.append(text)
                list2.append(indication)
                    
               

           if hanzi in dict1["cotton_ball"]:
             
              txt=open(os.path.join(text_root,"cotton_ball.txt"),mode="r+",encoding="utf-8")
              ind=open(os.path.join(text_root,"cotton_ball_indication.txt"),mode="r+",encoding="utf-8")
              indicate=ind.readlines()
              t=txt.readlines()
              indication=''.join(hint.strip('\n\r\t\v') for hint in indicate)
              text=''.join(element.strip('\n\r\t\v') for element in t)
              for i in range(5):
                img=Image.open(os.path.join(image_root,filename)).convert("RGB")
                inputs =processor(images=img)
                with  torch.no_grad():                      
                    image_features =model.get_image_features(**inputs)
                    img_features.append(image_features)
                list1.append(text)
                list2.append(indication)
                
       
       list1=[text  for text in list1 if  text.strip()]       
       lst2=[x for x  in list2 if x.strip()]
       t5_embed_dim=config1['encoder-decoder_embed_dim'][sequence]
       text_feature=np.array(tokenizer1(list1, return_tensors="pt", padding=True, truncation=True)['input_ids'])
       indication_feature=np.array(tokenizer1(lst2, return_tensors="pt", padding=True, truncation=True)['input_ids'])
       new_indication=np.zeros((indication_feature.shape[0],t5_embed_dim),dtype=np.float32)
       
       new_output=np.zeros((text_feature.shape[0],t5_embed_dim),dtype=np.float32)
       
      # For the purpose of aligning dimension of different_sentence indications and avoid diff of dim1
       if text_feature.shape[0]>1:
        for i in range(text_feature.shape[0]):
         len =len(indication_feature[i])
         len1=len(text_feature[i])
         len1_list.append(len1)
         len_list.append(len)
        max_len=max(len_list)
        text_max_len=max(len1_list)
        new_indication[:indication_feature.shape[0],:max_len]=indication_feature[:,:]
        new_output[:text_feature.shape[0],:text_max_len]=text_feature[:,:]
       else:
        max_len=indication_feature.shape[0]
        text_max_len=text_feature.shape[0]
        new_indication[:,:max_len]=indication_feature[:,:]
        new_output[:,:text_max_len]=text_feature[:,:]
      
       
       np.savez_compressed("J:/image_language_model/image_text.npz",all=np.array(image_features),indication=indication_feature,output=list(new_output),text=list(new_indication),image_path=list3,allow_pickle=True)
       
def preprocess_image_custom(image_list, size=512):
    transform = transforms.Compose([
        transforms.Resize((size, size)),  # 调整输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4814, 0.4578, 0.4060], 
                           std=[0.2686, 0.2613, 0.2758])  # ImageNet标准化
    ])
    return [transform(image) for image in image_list]


   
def main():
     text_root=""
     image_root=""
     output_folder=""
     bg_color=(128,156,200)
     config1_path=""
     config2_path=""
     img_sequence=1
     sequence=0
     classify(text_root,image_root,output_folder,bg_color,config1_path,config2_path,img_sequence,sequence)
     
      
if __name__ =='__main__':
   main()
     
