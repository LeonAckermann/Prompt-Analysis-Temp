import os
#import shutil
import shutil
import torch

all_model_prompt = os.listdir("model")

all_model_prompt = [dir for dir in all_model_prompt if ".py" not in dir]


for dataset_file in all_model_prompt:
    #if "T5" not in dataset_file or "Small" in dataset_file:
    #if "RobertaLarge" not in dataset_file or "Small" in dataset_file:
    #    continue
    #if "Small" not in dataset_file or "sam" not in dataset_file:
    if "T5Large" not in dataset_file:
        continue

    #if dataset_file != "QQPPromptRoberta":
    #    continue


    #if dataset_file != "ethicscommonsensePromptRoberta":
    #if dataset_file != "MRPCPromptRoberta":
    #    continue

    #print(file)

    original_dir = "model/"+str(dataset_file)
    if os.path.isdir(original_dir):
        pass
    else:
        continue

    check_list = [file for file in os.listdir(original_dir) if "_task_prompt" in file]
    if len(check_list) == 0:
        continue

    ##:mean do not use

    ##Choose epoch
    max_epoch = 0

    #Haven't done


    #tweet #training (68.XX)
    #ethicsdeontologyPromptT5 (63.8)
    #ethicsjusticePromptT5 (60.XX)
    #QQP (86.6)
    #squadPromptT5 (62.7)
    #nq_openPromptT5
    #multi_newsPromptT5
    #samsumPromptT5

    #MNLI #training
    #snli #training


    if dataset_file == "IMDBPromptRoberta":
        max_epoch = 23
    elif dataset_file == "IMDBPromptRobertaSmall":
        max_epoch = 29
    elif dataset_file == "IMDBPromptRobertaLarge":
        max_epoch = 27
    elif dataset_file == "IMDBPromptRoberta_label":
        max_epoch = 40
    elif dataset_file == "IMDBPromptBert":
        max_epoch = 21
    elif dataset_file == "IMDBPromptT5":
        max_epoch = 70
    elif dataset_file == "IMDBPromptT5Small":
        max_epoch = 69 #45
    elif dataset_file == "IMDBPromptT5Large":
        max_epoch = 60

    elif dataset_file == "SST2PromptRoberta":
        max_epoch = 25
    elif dataset_file == "SST2PromptRobertaSmall":
        max_epoch = 38
    elif dataset_file == "SST2PromptRobertaLarge":
        max_epoch = 26
    elif dataset_file == "SST2PromptRoberta_label":
        max_epoch = 18
    elif dataset_file == "SST2PromptBert":
        max_epoch = 18
    elif dataset_file == "SST2PromptT5":
        max_epoch = 26
    elif dataset_file == "SST2PromptT5Small":
        max_epoch = 17 #9
    elif dataset_file == "SST2PromptT5Large":
        max_epoch = 36

    elif dataset_file  == "laptopPromptRoberta":
        max_epoch = 32
    elif dataset_file  == "laptopPromptRobertaSmall":
        max_epoch = 40
    elif dataset_file  == "laptopPromptRobertaLarge":
        max_epoch = 93
    elif dataset_file  == "laptopPromptRoberta_label":
        max_epoch = 32
    elif dataset_file  == "laptopPromptBert":
        max_epoch = 30
    elif dataset_file  == "laptopPromptT5":
        max_epoch = 210
    elif dataset_file  == "laptopPromptT5Small":
        max_epoch = 229 #92
    elif dataset_file  == "laptopPromptT5Large":
        max_epoch = 499

    elif dataset_file == "restaurantPromptRoberta":
        max_epoch = 33
    elif dataset_file == "restaurantPromptRobertaSmall":
        max_epoch = 50
    elif dataset_file == "restaurantPromptRobertaLarge":
        max_epoch = 126
    elif dataset_file == "restaurantPromptRoberta_label":
        max_epoch = 32
    elif dataset_file == "restaurantPromptBert":
        max_epoch = 31
    elif dataset_file == "restaurantPromptT5":
        max_epoch = 276
    elif dataset_file == "restaurantPromptT5Small":
        max_epoch = 224 #162
    elif dataset_file == "restaurantPromptT5Large":
        max_epoch = 100

    elif dataset_file == "movierationalesPromptRoberta":
        max_epoch = 21
    elif dataset_file == "movierationalesPromptRobertaSmall":
        max_epoch = 31
    elif dataset_file == "movierationalesPromptRobertaLarge":
        max_epoch = 62
    elif dataset_file == "movierationalesPromptRoberta_label":
        max_epoch = 48
    elif dataset_file == "movierationalesPromptBert":
        max_epoch = 24
    elif dataset_file == "movierationalesPromptT5":
        max_epoch = 197
    elif dataset_file == "movierationalesPromptT5Small":
        max_epoch = 373 #299
    elif dataset_file == "movierationalesPromptT5Large":
        max_epoch = 100

    elif dataset_file == "tweetevalsentimentPromptRoberta":
        max_epoch = 28
    elif dataset_file == "tweetevalsentimentPromptRobertaSmall":
        max_epoch = 37
    elif dataset_file == "tweetevalsentimentPromptRobertaLarge":
        max_epoch = 54
    elif dataset_file == "tweetevalsentimentPromptRoberta_label":
        max_epoch = 23
    elif dataset_file == "tweetevalsentimentPromptBert":
        max_epoch = 21
    elif dataset_file == "tweetevalsentimentPromptT5":
        max_epoch = 18
    elif dataset_file == "tweetevalsentimentPromptT5Small":
        max_epoch = 32 #20 better
    elif dataset_file == "tweetevalsentimentPromptT5Large":
        max_epoch = 38


    elif dataset_file == "MNLIPromptRoberta":
        max_epoch = 44
    elif dataset_file == "MNLIPromptRobertaSmall":
        max_epoch = 13
    elif dataset_file == "MNLIPromptRobertaLarge":
        max_epoch = 10 ###
    elif dataset_file == "MNLIPromptRoberta_label":
        max_epoch = 30
    elif dataset_file == "MNLIPromptBert":
        max_epoch = 34
    elif dataset_file == "MNLIPromptT5":
        max_epoch = 5
    elif dataset_file == "MNLIPromptT5Small":
        max_epoch = 4 ##


    elif dataset_file == "QNLIPromptRoberta":
        max_epoch = 51
    elif dataset_file == "QNLIPromptRobertaSmall":
        max_epoch = 48
    elif dataset_file == "QNLIPromptRobertaLarge":
        max_epoch = 33 ###
    elif dataset_file == "QNLIPromptRoberta_label":
        max_epoch = 67
    elif dataset_file == "QNLIPromptBert":
        max_epoch = 41
    elif dataset_file == "QNLIPromptT5":
        max_epoch = 30
    elif dataset_file == "QNLIPromptT5Small":
        max_epoch = 11 ##

    elif dataset_file == "WNLIPromptRoberta":
        max_epoch = 755
    elif dataset_file == "WNLIPromptRoberta_label":
        max_epoch = 755
    elif dataset_file == "WNLIPromptBert":
        max_epoch = 754

    elif dataset_file == "snliPromptRoberta":
        max_epoch = 29
    elif dataset_file == "snliPromptRobertaSmall":
        max_epoch = 9
    elif dataset_file == "snliPromptRobertaLarge":
        max_epoch = 4 ###
    elif dataset_file == "snliPromptRoberta_label":
        max_epoch = 17
    elif dataset_file == "snliPromptBert":
        max_epoch = 32
    elif dataset_file == "snliPromptT5":
        max_epoch = 2
    elif dataset_file == "snliPromptT5Small":
        max_epoch = 2 #


    elif dataset_file =="RTEPromptRoberta":
        max_epoch = 250
    elif dataset_file =="RTEPromptRoberta_label":
        max_epoch = 250
    elif dataset_file =="RTEPromptBert":
        max_epoch = 249


    elif dataset_file == "QQPPromptRoberta":
         max_epoch =  22
    elif dataset_file == "QQPPromptRobertaSmall":
         max_epoch = 17
    elif dataset_file == "QQPPromptRobertaLarge":
         max_epoch = 8 ###
    elif dataset_file == "QQPPromptRoberta_label":
         max_epoch = 26
    elif dataset_file == "QQPPromptBert":
         max_epoch = 24
    elif dataset_file == "QQPPromptT5":
         max_epoch = 10
    elif dataset_file == "QQPPromptT5Small":
         max_epoch = 2 ###

    elif dataset_file == "MRPCPromptRoberta":
        max_epoch = 66
    elif dataset_file == "MRPCPromptRobertaSmall":
        max_epoch = 53
    elif dataset_file == "MRPCPromptRobertaLarge":
        max_epoch = 96
    elif dataset_file == "MRPCPromptRoberta_label":
        max_epoch = 30
    elif dataset_file == "MRPCPromptBert":
        max_epoch = 27
    elif dataset_file == "MRPCPromptT5":
        max_epoch = 199
    elif dataset_file == "MRPCPromptT5Small":
        max_epoch = 213


    elif dataset_file == "recastfactualityPromptRoberta":
        max_epoch = 21
    elif dataset_file == "recastfactualityPromptRoberta_label":
        max_epoch = 21
    elif dataset_file == "recastfactualityPromptBert":
        max_epoch = 20

    elif dataset_file == "recastpunsPromptRoberta":
        max_epoch = 36
    elif dataset_file == "recastpunsPromptRoberta_label":
        max_epoch = 36
    elif dataset_file == "recastpunsPromptBert":
        max_epoch = 35

    elif dataset_file == "recastverbcornerPromptRoberta":
        max_epoch = 35
    elif dataset_file == "recastverbcornerPromptRoberta_label":
        max_epoch = 35
    elif dataset_file == "recastverbcornerPromptBert":
        max_epoch = 34

    elif dataset_file == "recastnerPromptRoberta":
        max_epoch = 30
    elif dataset_file == "recastnerPromptRoberta_label":
        max_epoch = 18
    elif dataset_file == "recastnerPromptBert":
        max_epoch = 20

    elif dataset_file == "recastsentimentPromptRoberta":
        max_epoch = 58
    elif dataset_file == "recastsentimentPromptRoberta_label":
        max_epoch = 58
    elif dataset_file == "recastsentimentPromptBert":
        max_epoch = 57

    elif dataset_file == "recastmegaveridicalityPromptRoberta":
        max_epoch = 32
    elif dataset_file == "recastmegaveridicalityPromptRoberta_label":
        max_epoch = 32
    elif dataset_file == "recastmegaveridicalityPromptBert":
        max_epoch = 31

    elif dataset_file == "ethicscommonsensePromptRoberta":
        max_epoch = 96
    elif dataset_file == "ethicscommonsensePromptRoberta_label":
        max_epoch = 96
    elif dataset_file == "ethicscommonsensePromptBert":
        max_epoch = 95

    elif dataset_file == "ethicsdeontologyPromptRoberta":
        max_epoch = 63
    elif dataset_file == "ethicsdeontologyPromptRobertaSmall":
        max_epoch = 61
    elif dataset_file == "ethicsdeontologyPromptRobertaLarge":
        max_epoch = 79 #125 ###
    elif dataset_file == "ethicsdeontologyPromptRoberta_label":
        max_epoch = 77
    elif dataset_file == "ethicsdeontologyPromptBert":
        max_epoch = 14
    elif dataset_file == "ethicsdeontologyPromptT5":
        max_epoch = 101
    elif dataset_file == "ethicsdeontologyPromptT5Small":
        max_epoch = 52 #

    elif dataset_file == "ethicsjusticePromptRoberta":
        max_epoch = 29
    elif dataset_file == "ethicsjusticePromptRobertaSmall":
        max_epoch = 150
    elif dataset_file == "ethicsjusticePromptRobertaLarge":
        max_epoch = 68  #127 ###
    elif dataset_file == "ethicsjusticePromptRoberta_label":
        max_epoch = 63
    elif dataset_file == "ethicsjusticePromptBert":
        max_epoch = 15
    elif dataset_file == "ethicsjusticePromptT5":
        max_epoch = 141
    elif dataset_file == "ethicsjusticePromptT5Small":
        max_epoch = 48
    ##elif dataset_file == "ethicsvirtuePromptRoberta":
    ##    max_epoch = 21


    elif dataset_file == "squadPromptT5":
        max_epoch = 23
    elif dataset_file == "squadPromptT5Small":
        max_epoch = 11 ##
    elif dataset_file == "nq_openPromptT5":
        max_epoch = 15
    elif dataset_file == "nq_openPromptT5Small":
        max_epoch = 11 ##
    elif dataset_file == "multi_newsPromptT5":
        max_epoch = 21
    elif dataset_file == "multi_newsPromptT5Small":
        max_epoch = 16 ##
    elif dataset_file == "samsumPromptT5":
        max_epoch = 85
    elif dataset_file == "samsumPromptT5Small":
        max_epoch = 28 ###


    else:
        print("--------------------")
        print("Did not need to genertate this promt_emb:", dataset_file)
        print("--------------------")
        continue
        '''
        for file in os.listdir(original_dir):
            present_epoch = int(file.strip().split("_")[0])
            if present_epoch > max_epoch:
                max_epoch = present_epoch
        '''

    original_dir = original_dir+"/"+str(max_epoch)+"_task_prompt.pkl"



    try:
        parameters = torch.load(original_dir, map_location=lambda storage, loc: storage)
        prompt_emb = parameters["model"]
    except:
        print(dataset_file,"has no trained task_prompt.pkl at epoch",max_epoch)
        continue


    target_dir = "task_prompt_emb"+"/"+str(dataset_file)
    if os.path.isdir(target_dir):
        pass
    else:
        os.mkdir(target_dir)


    target_dir = target_dir+"/"+"task_prompt"

    torch.save(prompt_emb, target_dir)

    print("Save:", target_dir, " Done")

