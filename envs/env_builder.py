import yaml 
import os 
from abc import abstractmethod
from flask import request,Flask,render_template,redirect

app=Flask(__name__)
word=[]
def build_env(file_root,config_path,device,train_type):
 env_config=load_env_file(config_path)
 for file_name in os.listdir(file_root):
  if train_type ==all:
   if file_name in env_config["whole_data"]:
     f=open(os.path.join(file_root,file_name),mode="a+",encoding="utf-8")
     lines=f.readlines()
    

    

@app.route('/upload_image',methods=["POST"])
def  write_parameter():
  
    embed_dim=request.form.get('embed_dim')
    num_heads=request.form.get('num_heads')
    patch_size=request.form.get('kernel_size')
    image_channel=request.form.get('image_channel')
    stride=request.form.get('stride')
    lr=request.form.get
    num_transfer_layers=request.form.get('num_transfer_layers')
    config=load_env_file("K:/image_language_model/text_image.yaml")
    config["parameters"]["kernel_size"]=patch_size
    config["parameters"]["num_heads"]=num_heads
    config["parameters"]["stride"]=stride
    config["parameters"]["num_transfer_layers"]=num_transfer_layers
    config["parameters"]["image_channel"]=image_channel
    config["parameters"]["lr"]=lr

    with open("J:/image_language_model/image-text.yaml","a+",encoding="utf-8") as f:
      env_config=yaml.dump(config,f)
    return {"installation": config["parameters"]}
def load_env_file(file):
    with open(file, "r+",encoding="utf-8") as stream:
        env_config = yaml.load(stream,Loader=yaml.FullLoader)
    return env_config


@app.route('/insert_text',methods=["POST"])
def write_text_image_parameters():
    
    iterations=request.form.get("epochs")
    kernel_size=request.form.get("kernel_size")
    lr=request.form.get('lr')
    train_factor=request.form.get('train_factor')
    image_channel=request.form.get('init_channel')
    stride=request.form.get("stride")
    padding=request.form.get("padding")
    config=load_env_file("K:/image_language_model/text_image.yaml")
    config["parameters"]["iterations"]=iterations
    config["parameters"]["init_channel"]=image_channel
    config["parameters"]["kernel_size"]=kernel_size
    config["parameters"]["lr"]=lr
    config["parameters"]["train_factor"]=train_factor
    config["parameters"]["stride"]=stride
    config["parameters"]["padding"]=padding
    with open("K:/image_language_model/text_image.yaml",mode="a+") as f:
       yaml.dump(config,f)
    return {"installation": config["parameters"]}

#使用后端操作员控制的参数
#@app.route('/update_text',methods=["POST"])