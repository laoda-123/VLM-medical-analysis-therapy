import torch.nn  as nn
import torch 
from env_builder import load_env_file
import torch.optim.adam as adam
from  sentence_transformers import SentenceTransformer
from transformers import Blip2Model,Blip2Processor,AutoModelForSequenceClassification
from transformers import CLIPModel,CLIPProcessor,CLIPTokenizer
from transformers import AutoTokenizer,AutoProcessor,AutoModel
from transformers import AutoModelForTokenClassification,Blip2ForConditionalGeneration
from transformers import T5ForConditionalGeneration,T5EncoderModel, T5Tokenizer
import clip
import torch.nn.functional as F
def run_sentence_transformer(env_path,sequence:int):
    env_config=load_env_file(env_path)
    return SentenceTransformer(env_config["Long_sentence"][sequence])
def run_encoder_decoder(env_path,sequence:int):
    env_config=load_env_file(env_path)
    if type=="encoder-decoder":
     encode=T5EncoderModel.from_pretrained(env_config["encoder-decoder"][sequence])
     tokenizer=T5Tokenizer.from_pretrained(env_config['encoder-decoder'][sequence])
    return encode,tokenizer
     
def run_clip_transformer(env_path,sequence:int):
    env_config=load_env_file(env_path)
    model_name=env_config["image_tokenizer"][sequence]
    processor=CLIPProcessor.from_pretrained(model_name)
    model=CLIPModel.from_pretrained(model_name)
    tokenizer=CLIPTokenizer.from_pretrained(model_name)
    return  processor,model,tokenizer
def run_auto_transformer(env_path,squence):
    env_config=load_env_file(env_path)
    processor=AutoProcessor.from_pretrained(env_config["Bert"][squence])
    model=AutoModel.from_pretrained(env_config["Bert"][squence])
    tokenizer=AutoTokenizer.from_pretrained(env_config["Bert"][squence])
    return model,tokenizer


def run_Vit_base_transformer(env_path,squence:int):
    env_config=load_env_file(env_path)
    processor=AutoProcessor.from_pretrained(env_config["image_encoder"][squence])
    tokenizer=AutoTokenizer.from_pretrained(env_config["image_encoder"][squence])
    return processor,tokenizer

def run_sentence_chunk(env_path,sequence):
    env_config=load_env_file(env_path)
    model_name=env_config["sentence_piece"][sequence]
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForTokenClassification.from_pretrained(model_name)
    return  tokenizer,model
def run_multiple_sentence_transformer(env_path,sequence):
   config=load_env_file(env_path)
   model_name=config['multiple_sentence'][sequence]
   model=AutoProcessor.from_pretrained(model_name)
   processor=AutoModelForSequenceClassification.from_pretrained(model_name ,
            num_labels=3,  # 0: 并列, 1: 转折, 2: 因果
            output_attentions=True  # 启用注意力输出用于连接词定位
        )
   return model,processor

class  token_encoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.logit_scale = self.model.logit_scale  # 复用CLIP的温度系数
        
        # 冻结所有参数，仅作为教师模型提供知识
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    def forward(self, images, texts):
        """
        Args:
            images: 预处理后的图像张量 (batch_size, 3, 224, 224)
            texts: 原始文本列表,长度为batch_size
        Returns:
            image_features: 归一化后的图像特征 (batch_size, 512)
            text_features: 归一化后的文本特征 (batch_size, 512) 取特定的片段作为文本特征
            logits_per_image: 图像到文本的相似度得分 (batch_size, batch_size)
            text-output: 归一化过后的结果(batch_size,patch_size,embed dim)
        """
        # 文本token化（与原生CLIP对齐，max_length固定为77）
        token_ids = clip.tokenize(
            texts,context_length=77,truncate=False
        ).to(self.device)
        
        # 原生CLIP推理
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(token_ids)
        
        # 归一化+计算相似度
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits_per_image = image_features @ text_features.t() * self.logit_scale.exp()
        
        return image_features, text_features, logits_per_image