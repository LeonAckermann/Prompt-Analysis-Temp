import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from .modelling_roberta import RobertaForMaskedLM
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
except:
    tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")


class REPromptRoberta(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(REPromptRoberta, self).__init__()
        self.plmconfig = AutoConfig.from_pretrained("roberta-base")
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")


        #self.encoder = RobertaForMaskedLM.from_pretrained('../RobertaForMaskedLM/', config=self.plmconfig)
        self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        ##############
        ###Save a PLM + add prompt -->save --> load again
        #Build model and save it
        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
        else:
            from distutils.dir_util import copy_tree
            copy_tree("RobertaForMaskedLM/SST2PromptRoberta", self.init_model_path)
            os.remove(self.init_model_path+"/pytorch_model.bin")

            self.encoder = RobertaForMaskedLM.from_pretrained("roberta-base", config=self.plmconfig)
            torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            print("Save Done")

        ##############
        #self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)


        self.hidden_size = 768

        self.criterion = nn.CrossEntropyLoss()
        labelind = json.load(open(config.get("data", "label_index"), "r"))
        label2id = json.load(open(config.get("data", "label_info"), "r"))["label2id"]
        id2label = {label2id[l]: l for l in label2id}

        #print("======")
        #print(id2label)
        #print("------")
        #print(labelind)
        #print("======")

        self.labelindex = torch.tensor([labelind[id2label[i]][1] for i in range(len(id2label))], dtype=torch.long)

        #print(self.labelindex)
        #exit()

    def init_prompt_emb(self, init_ids):
        self.encoder.roberta.embeddings.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long).to(torch.cuda.current_device()))

    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output=False, **kwargs):
        if prompt_emb_output == True:
            output, prompt_emb = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len)
        else:
            output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'])

        logits = output["logits"] # batch, seq_len, vocab_size
        mask_logits = logits[:, 0] # batch, vocab_size
        score = mask_logits[:, self.labelindex.to(logits.device)]
        # score = torch.cat([mask_logits[:, 10932].unsqueeze(1), mask_logits[:, 2362].unsqueeze(1)], dim = 1)

        loss = self.criterion(score, data["label"])
        acc_result = acc(score, data['label'], acc_result)
        if prompt_emb_output==True:
            return {'loss': loss, 'acc_result': acc_result}, prompt_emb, data['label']
        else:
            return {'loss': loss, 'acc_result': acc_result}

def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())
    return acc_result
