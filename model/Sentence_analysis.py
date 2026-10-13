import torch 
from transformers import AutoTokenizer,AutoProcessor,AutoModelForTokenClassification,AutoModelForSequenceClassification
from envs.env_run import run_sentence_chunk
class LogicalRelationSplitter:
    def __init__(self,config3,sequence):
        # 加载预训练模型和Tokenizer
        model_name=config3['Bert'][sequence]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # 0: 并列, 1: 转折, 2: 因果
            output_attentions=True  # 启用注意力输出用于连接词定位
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 中文逻辑连接词词典
        self.conjunction_dict = {
            "并列": ["和", "与", "以及", "同时", "并且"],
            "转折": ["但是", "然而", "可是", "不过", "虽然","但是"],
            "因果": ["因为", "所以", "因此", "导致", "由于"]
        }
        self.En_conjunction_dict={"Parallelism":['and','as well as','align with','also','meanwhile'],
                                  "Turnaround":['but','nonetheless','however','unforunately','unluckily','conversely'],"Perhaps":['perhaps','maybe','could','probably','or']}
    def split_by_logical_relation(self, text):
        """
        输入：including logical sentences
        输出：sentence list relationship between different conjunction and logical relation
        """
        # 1. Tokenize输入文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 2. 模型推理获取逻辑关系分类和注意力权重
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 3. 解析模型结果
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1)# choose  label
        relation_types = ["并列", "转折", "因果"]
        predicted_relation = relation_types[predicted_label]
        
        # 4. 定位连接词（基于注意力权重）
        attentions = outputs.attentions[-1]  # 最后一层注意力
        cls_attention = attentions[0, :, 0, :].mean(dim=0)  # [CLS] token的平均注意力
        token_attn_scores = cls_attention.cpu().numpy()
        
        # 5. 找到注意力最高的连接词
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        candidate_conjunctions = self.conjunction_dict[predicted_relation]
        best_conjunction = None
        max_attn_score = 0
        
        for token_idx, token in enumerate(tokens):
            if token in candidate_conjunctions and token_attn_scores[token_idx] > max_attn_score:
                best_conjunction = token
                max_attn_score = token_attn_scores[token_idx]
        
        # 6. 切分句子
        if best_conjunction and best_conjunction in text:
            split_parts = text.split(best_conjunction)
            # 去除空字符串和多余空格
            split_parts = [part.strip() for part in split_parts if part.strip()]
            return {
                "original_text": text,
                "logical_relation": predicted_relation,
                "conjunction": best_conjunction,
                "split_sentences": split_parts
            }
        else:
            return {
                "original_text": text,
                "logical_relation": predicted_relation,
                "conjunction": None,
                "split_sentences": [text]
            }
class SentenceComponentAnalysis():
    def __init__(self,config,sequence): 
        
        self.tokenizer,self.model=run_sentence_chunk(config,sequence)
        self.model_config= self.model.config.id2label
        self.label_list=config['label_list']
    def analyze(self, sentence):
        # 模型推理
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        # 解析结果
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        results = []
        for token, pred in zip(tokens, predictions[0]):
            results.append((token, self.label_list[pred.item()]))
        
        # 提取主谓宾表成分
        components = self._extract_components(results)
        return results, components
    
    def _extract_components(self, results):
        components = {
            'subject': [],
            'predicate': [],
            'object': [],
            'attribute': []
        }
        
        for token, label in results:
            if label.startswith('B-NP') or label.startswith('I-NP'):
                # 判断是主语、宾语还是表语（简化处理，实际需要更复杂的规则）
                components['subject'].append(token)
            elif label.startswith('B-VP') or label.startswith('I-VP'):
                components['predicate'].append(token)
            elif label.startswith('B-ADJP') or label.startswith('I-ADJP'):
                components['attribute'].append(token)
        
        return components
    