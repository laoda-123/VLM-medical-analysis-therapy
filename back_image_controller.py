from flask import Flask,request,jsonify,render_template,send_file
from elasticsearch import Elasticsearch

from net.video_recognition import stable_diffusion
from net.value_critic_Net import StepNet

import  pymysql 

from envs.env_builder import load_env_file

from keybert import KeyBERT
from flask_socketio import SocketIO, emit
from PIL import Image

import torch
import io

import uuid
from envs.env_builder import load_env_file
import pytz
import time
from VitExtractor import extract_keyword
from net.image_recognition import image_transform
from net.image_text_critic import criticNet
from envs.env_run import run_encoder_decoder
from envs.env_run import run_clip_transformer
es=Elasticsearch(['https://www.so.com'])



app=Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def image_processed_callback():
    socketio.emit('image_updated', {'message': '图片已更新'}, namespace='/')



def get_image_text_parameters(image_text_config:str):
    config=load_env_file(image_text_config
                         )
    return config

@app.route('/text_input',methods=["POST"])

def get_post(config1_path, config2_path,sequence,pt_path): #config1 is image_description.yaml while config2 is text image.yaml
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    word_input=request.form.get('diff_input')
    config1,config2=load_env_file(config1_path),load_env_file(config2_path)
    encode,tokenizer=run_encoder_decoder(config1_path,sequence)
    
    model=stable_diffusion(config1,config2,sequence,device)
    model.eval()
    
    keywords=extract_keyword(word_input,top_n=6)
    word_tokenizer=tokenizer(word_input,
    padding="max_length",  # 填充到最大长度
    truncation=True,       # 过长文本截断
    max_length=512,        # T5模型最大输入长度
    return_tensors="pt"    # 返回PyTorch张量
    )
    model.load_state_dict(torch.load(pt_path))
    output_images=model(word_tokenizer)
    img=(output_images+1)/2*255

    img_io = io.BytesIO(img)
    img.save(img_io, 'PNG',quality=95)
    img_io.seek(0)  
    data={"image_type": keywords,"image":img_io}
    return send_file(img_io,mimetype='image/png',as_attachment=False,download_name='processed_image.png')
    

@app.route('/file-upload.html',methods=["POST"])
def get_file(config1_path,config2_path,sequence,img_sequence,pt_path): # config1_path is image_description.yaml while config2 is image-text.yaml
    config1,config2=load_env_file(config1_path),load_env_file(config2_path)
    processor,model,tokenizer=run_clip_transformer(config1_path,img_sequence)
    encode,tokenizer=run_encoder_decoder(config1_path,sequence)
    prompts=request.form.get('user_input')
    input_ids=tokenizer(prompts,return_tensors="pt", padding=True, truncation=True)
    
    
    start_time=time.monotonic()
    try: 
     connection=pymysql.connect(host='127.0.0.1',user='root',password='5201314jntm',database='mamba',port=3306, charset='utf8mb4')
    except pymysql.MySQLError as e:
        print(f"数据库连接失败: {str(e)}")
        return jsonify({'status': 'failed','message': '404 Not Found'})
    images=request.files.getlist('/file-upload.html')
    image_list,output_table=[],[]
    
    for image in images:
      image=Image.open(image.stream).convert("RGB")
      inputs=processor(image)
      with torch.no_grad():
          image_features=model.get_image_features(**inputs)
      image_list.append(image_features)
    image_tensor=torch.stack(image_list,dim=0)

    table=connection.cursor()
    table.execute("SELECT DISTINCT * AS  non_null_rows  FROM train_id ")
    output_table=[row[0] for  row in table.fetchall()]
    output_kinds=len(output_table)
    
    keyword=[]
    
    image_tran=image_transform(config1,config2,sequence)
    image_tran.eval() 
    image_tran.load_state_dict(torch.load(pt_path))
    type_output=image_tran(image_tensor,input_ids)
    decoder_text=tokenizer.decode(type_output)
    

     
    if not es.indices.exists(index='documents'):
     es.indices.create(
        index='documents',
        body={
            'settings': {
                'analysis': {
                    'analyzer': {
                        'ik_max_word': {
                            'type': 'ik_max_word'
                        }
                    }
                }
            },
            'mappings': {
                'properties': {
                    'title': {'type': 'text', 'analyzer': 'ik_max_word'},
                    'content': {'type': 'text', 'analyzer': 'ik_max_word'},
                    'url': {'type': 'keyword'}
                }
            }
        }
    )
    body={
        'query': {
            'multi_match': {
                'query': keyword,
                'fields': ['title^2', 'content'],  # title权重更高
                'analyzer': 'ik_max_word'
            }
        },
        'highlight': {
            'fields': {
                'title': {},
                'content': {}
            }
        }
    }
    response = es.search(index='documents', body=body)
    total = response['hits']['total']['value']
    results = []

    for hit in response['hits']['hits']:
        # 处理高亮结果
        title = hit['highlight']['title'][0] if 'title' in hit['highlight'] else hit['_source']['title']
        content = hit['highlight']['content'][0] if 'content' in hit['highlight'] else hit['_source']['content'][:200] + '...'
    results.append({
            'title': title,
            'content': content,
            'url': hit['_source']['url'],
            'score': hit['_score']
        })
    results.append(decoder_text)
    render_template('/output_text.html',
        data=results,message="Your input value is below"
    )
    end_time=time.monotonic()
    date_time=end_time-start_time
    return {"status": 'success',"counting":date_time}
@app.route('/download_user_image', methods=["POST"])
def get_save_path():
      save_path=request.form.get('dirNameEl')
      if save_path is not None:
          return {"save_path":save_path}
      else:
          return jsonify({"status":"failed",'message':'401 Not got'})

