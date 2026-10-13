from flask import request,Flask,render_template,Response,app,jsonify
import torch 
import os 
import pymysql
from envs.pretrain_processing import get_text_image_input
from sqlalchemy import Table, Column, MetaData
from net.image_recognition import cal_sim
from PIL import Image
import numpy as np 
app=Flask(__name__) 
app.config['STATIC_FOLDER']="H:/static_cache"
image_root="https://pixabay.com/photos"
meta=MetaData()
def create_db_table(table_name:str):
    conn=pymysql.connect(host="127.0.0.1" ,user="root",passwd="5201314jntm",db="mamba")
    table=conn.cursor()
    table.execute("CREATE TABLE [IF NOT EXISTS] train_text ( id INT UNIQUE NOT NULL AUTO_INCREMENT, text  VARCHAR(45) UNIQUE NOT NULL  PRIMARY_KEY, \
         image_path VARCHAR(400) NOT NULL  ) ")
    
@app.route('/insert-text',methods=["POST"])
def get_parameter():
    keywords_dict=get_text_image_input()
    keywords=keywords_dict["keyword"]
    files=request.files.getlist('fileInput')
    conn=pymysql.connect(host="127.0.0.1",user="root",passwd="5201314ujntm",db="mamba")
    table=conn.cursor()
    dir_dict=get_download_path()
    dirname=dir_dict['message']
    arr_list=[]
    for file in files:
        filename=file.name 
        img=file.stream
        img_arr=np.array(img)
        if filename in os.listdir(image_root):
         continue
        else:
         arr_list.append(img_arr)
    table.execute("SELECT DISTINCT text AS  non_null_rows FROM train_text " )
    kw_list=table.fetchall() 
    keyword_list=[word[0] for  word in kw_list]
    length=len(keyword_list)
    table.execute("INSERT INTO train_text(id,text,image_path) VALUES(len+1,keywords,filename)")

    if dirname is not None:
      np.savez_compressed(dirname,keyword=keyword_list, table=torch.tensor(arr_list))
    table.close()
    conn.close()
    return jsonify({"status":"success","message": None})
@app.route('/delete-types',methods=["POST"])
def delete_img():
   keywords=request.form.get('delete')
   conn=pymysql.connect(host="127.0.0.1",user="root",passwd="5201314jntm",db="mamba")
   table=conn.cursor()
   table.execute("SELECT DISTINCT text AS  non_null_rows FROM train_text " )
   keyword_list=table.fetchall() 
   dir_dict=get_download_path()
   dirname=dir_dict['message']
   for keyword in keyword_list:
      if cal_sim(keywords,keyword)>0.90:
         table.execute("DELETE * FROM train_text WHERE keywords==keyword")
      else:
         print(f"Sorry,we can't find the wanted delete words")
   keyword_list=table.execute("SELECT DISTINCT text AS  non_null_rows FROM train_text " )
   filepath=table.execute("SELECT DISTINCT image_path AS  non_null_rows FROM train_text " )
   filepath=[]
   img_arr_list=[]
   for file in filepath:
      img=Image.open(file).convert("RGBA")
      img_arr_list.append(img)
   if dirname is not None:
      np.savez_compressed(dirname,keyword=keyword_list, table=torch.tensor(img_arr_list))
   table.close()
   conn.close()
   return jsonify({"status": 'success',"message": None})
@app.route('/update-types',methods=["POST"])
def update_img():
   imgs=request.files.getlist('update')
   conn=pymysql.connect(host="127.0.0.1",user="root",passwd="5201314jntm",db="mamba")
   table=conn.cursor()
   save_dict=get_download_path()
   dirname=save_dict['message']
   table.execute("SELECT DISTINCT image_path AS  non_null_rows FROM train_text")
   for img  in imgs:
     img1=img.stream
     img1=img1.read()
     for filename in [row[0]  for row in table.fetchall()] :

      filepath=os.path.join(image_root, filename)
      img2=Image.open(filepath).convert("RGBA")
      if os.path.exists(filepath):
       return render_template('/train_text_input.html',message="You have put the same image")
      else:
       if cal_sim(img1,img2)>0.95:
        keyword=table.execute("SELECT keyword from train_text WHERE image_path==filename")
        len=len(keyword)
        table.execute("INSERT INTO  train_text(id,text,image_path) VALUES(len+1,keyword,img.name)")
   img_arr_list=[]
   keyword_list=table.execute("SELECT DISTINCT text AS  non_null_rows FROM train_text " ) 
   list_path=table.execute("SELECT DISTINCT image_path AS  non_null_rows FROM train_text")
   paths=[row[0] for row in table.fetchall()]

   for path in paths:
     img=img.open(os.path.join(image_root,path)).convert("RGBA")
     img_arr_list.append(img)
   np.savez_compressed(dirname, keywords=keyword_list,target_result=torch.tensor(img_arr_list))
   table.close()
   conn.close()
   return jsonify({"status":'success',"message":None})
@app.route('/download_npz',methods=["POST"])
def get_download_path():
  dirname=request.form.get('dirNameEl')
  return {"status":'success','message': dirname}
