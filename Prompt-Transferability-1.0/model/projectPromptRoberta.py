import torch
import torch.nn as nn
import torch.nn.functional as F
import json


import os
import datasets

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from .modelling_roberta import RobertaForMaskedLM
#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
except:
    tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")


#{0: 'imdb', 1: 'laptop', 2: 'mnli', 3: 'mrp', 4: 'qnli', 5: 'qqp', 6: 're', 7: 'restaurant', 8: 'rte', 9: 'sst2', 10: 'stsb', 11: 'wnli'}

def load_task_prompt(config):
    #choosed_tasks=['imdb','laptop','mnli','mrp','qnli','qqp','re','restaurant','rte','sst2','stsb','wnli']
    #choosed_tasks=['imdb','laptop','mnli','mrp','qnli','qqp','restaurant','rte','sst2','wnli']

    #choosed_tasks=['imdb','laptop','mrp','qqp','restaurant','sst2','wnli']

    if "bert" in config.get("model","model_base").lower():
        model_prompt = "Bert"
    elif "roberta" in config.get("model","model_base").lower():
        model_prompt = "Roberta"



    choosed_tasks = config.get("data","train_dataset_type").lower().split(",")

    name_list = list()
    task_prompt_dict=dict()
    task_prompt_ten=list()
    path="./task_prompt_emb"
    files = os.listdir(path)
    for file in files:
        #print(file)
        if "proj" in file or "mlm" in file:
            continue
        task_prompt_emb = torch.load(path+"/"+file+"/task_prompt")
        name = str(file.strip().split("P")[0]).lower()
        if name=="mr":
            name+="pc"
        elif name=="qq":
            name+="p"
        if name not in choosed_tasks or model_prompt in file:
            continue
        #print(name,file)

        #print(task_prompt_emb.shape)
        #print(name)
        print(file)
        name_list.append(name)
        task_prompt_dict[name] = task_prompt_emb
    #print(name_list)
    name_list.sort()
    #name_dict = {id:n for id,n in enumerate(name_list)}
    #print(name_dict)
    #exit()

    print("======")
    print("map:",name_list)
    print("======")
    #exit()

    #for id, name in name_dict.items():
    for id, name in enumerate(name_list):
        task_prompt_ten.append(task_prompt_dict[name].to("cuda"))
    task_prompt_ten = torch.stack(task_prompt_ten).to("cuda")

    return task_prompt_ten


class projectPromptRoberta(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        #super(PromptRoberta, self).__init__()
        super(projectPromptRoberta, self).__init__()


        try:
            if config.get("model","model_size")=="large":
                model = "roberta-large"
                ckp = "RobertaLargeForMaskedLM"
                self.hidden_size = 1024
            else:
                model = "roberta-base"
                ckp = "RobertaForMaskedLM"
                self.hidden_size = 768
        except:
            model = "roberta-base"
            ckp = "RobertaForMaskedLM"
            self.hidden_size = 768

        self.task_specific_prompt_emb = load_task_prompt(config).to('cuda')

        self.plmconfig = AutoConfig.from_pretrained(model)
        # self.plmconfig["architectures"] = ["RobertaForMaskedLM"]
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")
        #self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        #self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        self.init_model_path = str(ckp)+"/"+config.get("data","train_formatter_type")
        ##############
        ###Save a PLM + add prompt -->save --> load again
        #Build model and save it
        #print(self.init_model_path)
        #exit()
        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
        else:
            from distutils.dir_util import copy_tree
            #copy_tree("RobertaForMaskedLM/SST2PromptRoberta", self.init_model_path)
            copy_tree(str(str(ckp)+"/SST2PromptRoberta"), self.init_model_path)
            os.remove(self.init_model_path+"/pytorch_model.bin")

            self.encoder = RobertaForMaskedLM.from_pretrained(model, config=self.plmconfig)
            torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            #torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            print("Save Done")

        ##############
        #self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)


        # self.encoder = AutoModelForMaskedLM.from_pretrained("roberta-base")
        #self.hidden_size = 768
        # self.fc = nn.Linear(self.hidden_size, 2)
        if config.get("data", "train_dataset_type") == "STSB":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        # self.prompt_num = config.getint("prompt", "prompt_len") # + 1
        # self.init_prompt_emb()

        #Refer to https://github.com/xcjthu/prompt/blob/master/model/PromptRoberta.py : line31 revised
        #self.labeltoken = torch.tensor([10932, 2362], dtype=torch.long)
        #self.softlabel = config.getboolean("prompt", "softlabel")
        #if self.softlabel:
        #    self.init_softlabel(self.plmconfig.vocab_size, len(self.labeltoken))



        self.random_init_prompt = nn.Embedding(int(self.plmconfig.prompt_num),int(self.hidden_size))
        self._init_weights(self.random_init_prompt)
    def return_init_prompt_emb_(self, module):
        #self.random_init_prompt = nn.Embedding(int(config.prompt_num),int(config.hidden_size))
        self._init_weights(self.random_init_prompt)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



    def init_prompt_emb(self, init_ids):
        self.encoder.roberta.embeddings.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long).to(torch.cuda.current_device()))


    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output="replace_task_specific_prompt_emb", **kwargs):
        # print(self.encoder.roberta.embeddings.prompt_embeddings.weight)
        if prompt_emb_output == True:
            output, prompt_emb = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len)
        elif prompt_emb_output == "replace_task_specific_prompt_emb":
            task_specific_prompt_emb = torch.index_select(self.task_specific_prompt_emb, 0, data["task_name"])

            model_AE = kwargs["AE"]


            task_specific_prompt_emb_ = task_specific_prompt_emb.reshape(int(task_specific_prompt_emb.shape[0]), int(task_specific_prompt_emb.shape[1])*int(task_specific_prompt_emb.shape[2]))
            task_specific_prompt_emb_ = model_AE(task_specific_prompt_emb_).to("cuda")
            task_specific_prompt_emb = task_specific_prompt_emb_.reshape(int(task_specific_prompt_emb.shape[0]),int(task_specific_prompt_emb.shape[1]),int(task_specific_prompt_emb.shape[2]))


            #print("======")
            #print(self.random_init_prompt)
            #print("======")


            output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len, task_specific_prompt_emb=task_specific_prompt_emb)
        else:
            output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'])

        # batch, seq_len = data["inputx"].shape[0], data["inputx"].shape[1]
        # prompt = self.prompt_emb.weight # prompt_len, 768

        # input = self.encoder.get_input_embeddings()(data["inputx"])
        # embs = torch.cat([prompt.unsqueeze(0).repeat(batch, 1, 1), input], dim = 1)

        # output = self.encoder(attention_mask=data['mask'], inputs_embeds=embs)


        logits = output["logits"] # batch, seq_len, vocab_size #torch.Size([16, 231, 50265])

        mask_logits = logits[:, 0] # batch, vocab_size #torch.Size([16, 50265])


        '''
        print("==============")
        print("==============")

        #sentiment
        #mo_dict={"positive":0,"neutral":1,"negative":2,"conflict":3}
        print(tokenizer.encode("positive",add_special_tokens=False)) #22173
        #print(tokenizer.encode("neutral",add_special_tokens=False)) #12516
        print(tokenizer.encode("moderate",add_special_tokens=False)) #19397
        print(tokenizer.encode("negative",add_special_tokens=False)) #33407
        print(tokenizer.encode("conflict",add_special_tokens=False)) #'conf':17075,, 'lict':

        #NLI
        print(tokenizer.convert_ids_to_tokens([10932])) #['yes']
        print(tokenizer.convert_ids_to_tokens([12516])) #['neutral']
        print(tokenizer.convert_ids_to_tokens([2362])) #['no']

        #paraphrase
        print(tokenizer.encode("true",add_special_tokens=False)) #[29225]
        print(tokenizer.encode("false",add_special_tokens=False)) #[22303]

        print(tokenizer.encode("right",add_special_tokens=False)) #[4070]
        print(tokenizer.encode("wrong",add_special_tokens=False)) #[35621]

        '''

        #label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:postive, 8:conflict}
        #score = torch.cat([mask_logits[:,2362].unsqueeze(1), mask_logits[:,10932].unsqueeze(1), mask_logits[:,22303].unsqueeze(1), mask_logits[:,12516].unsqueeze(1),mask_logits[:,29225].unsqueeze(1),mask_logits[:,33407].unsqueeze(1),mask_logits[:, 19397].unsqueeze(1),mask_logits[:,22173].unsqueeze(1),mask_logits[:,17075].unsqueeze(1)], dim=1)

        #label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:postive, 8:conflict, 9:low, 10:high}
        score = torch.cat([mask_logits[:,2362].unsqueeze(1), mask_logits[:,10932].unsqueeze(1), mask_logits[:,22303].unsqueeze(1), mask_logits[:,12516].unsqueeze(1),mask_logits[:,29225].unsqueeze(1),mask_logits[:,33407].unsqueeze(1),mask_logits[:, 19397].unsqueeze(1),mask_logits[:,22173].unsqueeze(1),mask_logits[:,17075].unsqueeze(1), mask_logits[:,5481].unsqueeze(1), mask_logits[:,3530].unsqueeze(1)], dim=1)

        '''
        if config.get("data", "train_dataset_type") == "laptop" or config.get("data", "train_dataset_type") == "restaurant" :
            #sentiment
            #mo_dict={"positive":22173,"moderate":19397,"negative":33407,"conflict":17075}
            score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:, 19397].unsqueeze(1), mask_logits[:, 22173].unsqueeze(1), mask_logits[:,17075].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "SST2" or config.get("data", "train_dataset_type") == "IMDB":
            #sentiment
            #mo_dict={"positive":22173,"negative":33407}
            score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:,22173].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "MNLI":
            #NLI
            #mo_dict={"yes":10932,"neutral":12516,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 12516].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "RTE":
            #NLI
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "WNLI":
            #NLI
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "QNLI":
            #NLI
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "MRPC":
            #paraphrase
            #mo_dict={"true":29225,"false":22303}
            score = torch.cat([mask_logits[:, 22303].unsqueeze(1), mask_logits[:,29225].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "QQP":
            #paraphrase
            #mo_dict={"true":29225,"false":22303}
            score = torch.cat([mask_logits[:, 22303].unsqueeze(1), mask_logits[:,29225].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "STSB":
            score = mask_logits[:, 10932]
        else:
            #Other
            #mask_logits:torch.Size([16, 50265])
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        '''



        loss = self.criterion(score, data["label"])
        #if config.get("data", "train_dataset_type") == "STSB":
        #    acc_result = pearson(score, data['label'], acc_result)
        #else:
        acc_result = acc(score, data['label'], acc_result)

        if prompt_emb_output == True:
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


def pearson(score, label, acc_result):
    stsb_result = cal_pearson(score, label)
    if acc_result is None:
        acc_result = {'total_pearson': 0, 'batch_num': 0}
    acc_result['total_pearson'] += stsb_result['pearson']
    acc_result['batch_num'] += 1
    return acc_result


def cal_pearson(score, label):
    tmp_result = {}
    score_bar = torch.mean(score, dim=-1)
    label_bar = torch.mean(label, dim=-1)
    numerator = torch.sum(torch.mul(score-score_bar, label - label_bar), dim=-1)
    denominator = torch.sqrt(torch.sum((score-score_bar) ** 2, dim=-1)) * torch.sqrt(torch.sum((label-label_bar) ** 2, dim=-1))
    pearson_result = numerator / denominator
    tmp_result['pearson'] = pearson_result.item()
    return tmp_result
