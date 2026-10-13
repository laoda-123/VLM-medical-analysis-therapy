from multiprocessing.dummy import connection
from flask import Flask,request,jsonify,render_template, redirect, url_for
import pymysql

from classify_data import batch_rgba_convert_rgb
from classify_data import watershed_contour_extract
import torch
import numpy  as np
from PIL import Image
import os
from pathlib import Path
from  abc import abstractmethod
from flask_wtf.csrf import CSRFProtect
import re
from envs.env_builder import load_env_file

app=Flask(__name__) 
app.config['SECRET_KEY'] = '5201314jntm?'  # 必须设置，用于加密CSRF令牌
csrf = CSRFProtect(app)


env_config=load_env_file("J:/image_language_model/image_root.yaml")
image_root=env_config["image_com"][0]

@app.route('/upload_image',methods=["POST"])
def put():
  images=request.files.getlist("user_image")
  class_in=request.form.get("classification")
  exit_or_not=request.form.get("agree_to_exit_or_not")

  img=[ ]
  
  connection=pymysql.connect(host="127.0.0.1",user="root",passwd="5201314jntm",db="mamba")
  table=connection.cursor()
  table.execute("SELECT COUNT(index) AS non_null_rows FROM train_id")
  length=len([row[0] for row in table.fetchall()])
  if table.execute("SELECT EXISTS (SELECT class_in FROM train_id WHERE classification==class_in)"):
    length=length
  else:
    length+=1
    table.execute("INSERT INTO train_id(index, classification) VALUES(len,class_in)")
  img_arr_list=[]
  types=[]
  table.execute("SELECT DISTINCT classification AS  non_null_rows FROM train_id ")
  names=[row[0] for row in table.fetchall()]
  for image in images:
   if image.name  in os.listdir(image_root):
    continue
   uploadPath=os.path.join(image_root,image.name)
   image.save(uploadPath)
   watershed_contour_extract(uploadPath,uploadPath)
   img=batch_rgba_convert_rgb(uploadPath,uploadPath)
   for filename in os.listdir(image_root):
      class_input=process_filepath(filename)
      type=re.findall(r"[a-zA-Z]+",class_input)
      types.append(type)
      img_arr=np.array(img)
      img_arr_list.append(img_arr)
  length=len(names)
  arr=np.zeros((length,))
  output=[] 
  for type in types:
    index=names.index(type)
    arr[index]=1
    output.append(arr)
      

  download_dict=get_path()
  downloadPath=download_dict["data"]
  
  if os.path.isdir(downloadPath):
   np.savez_compressed(downloadPath,type=types,img_arr=img_arr_list,tensor=torch.tensor(output))
  json=get_exit()
  if json.get_data()=="exit" :
    table.close()
    connection.close()
  return   jsonify({"status": 'success'})
  
@app.route('/delete-image',methods=["POST"])
def delete():
   text=request.form.get("delete_image_type")
   types=[]
   img_arr_list=[]
   output=[]
   if text is  None:
     print(jsonify({"status":False,"text":None}))
   conn=pymysql.connect(host="127.0.0.1",user="root",password="5201314jntm",database='mamba')
   table=conn.cursor()
   
   try:
    
     for filename in os.listdir(image_root):
       file=process_filepath(filename)
       type=re.findall(r"[a-zA-Z]+",file)
       types.append(type)
       
       if text in types:
         table.execute("DELETE * FROM train_id WHERE classification==text")
         table.fetchall()
         os.remove(os.path.join(image_root,filename))
         len-=1
         types.remove(text)
       filepath=os.path.join(image_root,filename)
       img=Image.open(filepath).convert("RGBA")
       img_arr=np.array(img)
       img_arr_list.append(img_arr)
     else:
         len=len
   except Exception as e:
     print(f"sorry, your deleted type can't be found or you may not include your image in the dataset")
   table.execute("SELECT DISTINCT classification AS  non_null_rows FROM train_id ")
   names=[row[0] for row in table.fetchall()]
  
   length=len(names)
   arr=np.zeros((length,))
   for type in types:
     index=names.index(type)
     arr[index]=1.0 
     output.append(arr)

   downloadPath=get_path()['data']
   if os.path.isdir(downloadPath):
      np.savez_compressed(downloadPath,type=types,img_arr=img_arr_list,tensor=torch.tensor(output))
  
   mess=get_exit()
   if mess.get_data()=="exit" :
    table.close()
    conn.close()
   return jsonify({"status": 'success','message':None})
@app.route('/update-classification',methods=["POST"])
def update():
  files=request.files.getlist("update-file")
  delete_add=request.form.get("add_delete_img")
  types=[]
  conn=pymysql.connect(host='127.0.0.1',user='root',passwd="5201314jntm",db="mamba")
  table=conn.cursor()
  table.execute("SELECT DISTINCT classification AS  non_null_rows FROM train_id ")
  names=[row[0] for row in table.fetchall()]
  target_result=[]
  arr_img_list=[]
  leibie=np.zeros((len(names),))
  for file in files:
    filename=file.name
    class_in=re.findall(r"[a-zA-z]",process_filepath(filename))
    if os.path.exists(os.path.join(image_root,filename)) and delete_add=="delete":
      os.remove(os.path.join(image_root,filename))

    if delete_add=="add" and os.path.exists(os.path.join(image_root,filename))==False:
      filepath=os.path.join(image_root,filename)
      file.save(filepath)
    img=Image.open(file.stream).convert("RGB")

    watershed_contour_extract(filepath,filepath)
    img=batch_rgba_convert_rgb(filepath,filepath)
  for  filename in os.listdir(image_root):
    type=re.findall(r"[a-zA-z]",process_filepath(filename))
    types.append(type)
    img=Image.open(os.path.join(image_root,filename)).convert("RGB")
    tensor=torch.tensor(img)
    arr_img_list.append(tensor)

  leibie=len(names)
  arr=np.zeros((leibie,))
  output=[]
  for type in types:
    index=names.index[type]
    arr[index]=1.0
    output.append(arr)
  json=get_exit()
  js=get_path()
  downloadPath=js["data"]
  if os.path.isdir(downloadPath):
   np.savez_compressed(downloadPath,type=types,img_arr=arr_img_list,tensor=output)
  if json.get_data()=="exit" :
    table.close()
    conn.close()
  return jsonify({"status":True, "message":"You have update your train image"})



@abstractmethod
@app.route('/download',methods=["POST"])
def get_path():
 download=request.form.get("download")
 save_path=request.form.get("dirNameEl")

 if os.path.exists(save_path) and download  is not None:
   return {"status":True,"data":save_path}
   
 else:
   return {"status":False,"data":"Your filepath doesn't exist"}
   

@abstractmethod
def process_filepath(filepath):
 return Path(filepath).stem

@abstractmethod
@app.route('/agree_to_exit_database',methods=["POST"])
def get_exit():
 exit_or_not=request.form.get('agree_to_exit_or_not')
 if exit_or_not=="Yes":
  return jsonify({"status":True,"data":"exit"})
 else:
  return jsonify({"status":False,"data":"stay"})