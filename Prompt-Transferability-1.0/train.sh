
#Download model checkpoints and move them to the corresponding dir


mkdir BertMediumForMaskedLM
mkdir BertForMaskedLM

mkdir RobertaForMaskedLM
mkdir RobertaLargeForMaskedLM

mkdir T5SmallForMaskedLM
mkdir T5ForMaskedLM
mkdir T5LargeForMaskedLM
mkdir T53BForMaskedLM
mkdir T511BForMaskedLM


# Bert: BertMedium, Bert (BertBase)
# Roberta: Roberta, RobertaBase (BertBase)
# T5: T5Small, T5 (T5Base), T5Large
gpus=7
DATASET="SST2"
BACKBONE="Roberta"

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/${DATASET}Prompt${BACKBONE}.config \
    --gpu $gpus \
    #--checkpoint model/${DATASET}Prompt${BACKBONE} \


'''
################################
###########Roberta##############
################################
############
#Sentiment
############
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed




#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed



#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \


#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \



############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \


#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \



############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \


###
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/emobankarousalPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/persuasivenessrelevancePromptRoberta.config \
    --gpu $gpus



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/persuasivenessspecificityPromptRoberta.config \
    --gpu $gpus




CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/emobankdominancePromptRoberta.config \
    --gpu $gpus



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/squinkyimplicaturePromptRoberta.config \
    --gpu $gpus




CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/squinkyformalityPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptRoberta.config \
    --gpu $gpus

##
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/anliPromptRoberta.config \
    --gpu $gpus



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRoberta.config \
    --gpu $gpus


#####Add tasks
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastfactualityPromptRoberta.config \
    --gpu $gpus



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastpunsPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastverbnetPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastverbcornerPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastnerPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastsentimentPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastmegaveridicalityPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicscommonsensePromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticePromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsvirtuePromptRoberta.config \
    --gpu $gpus



##########################################################
###########BERT###########################################
##########################################################

############
#Sentiment
############
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptBert.config \
    --gpu $gpus \



#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptBert.config \
    --gpu $gpus \



#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptBert.config \
    --gpu $gpus \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptBert.config \
    --gpu $gpus \



############
#Paraphrase
############


#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptBert.config \
    --gpu $gpus \



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptBert.config \
    --gpu $gpus \



############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptBert.config \
    --gpu $gpus \



#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptBert.config \
    --gpu $gpus \



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WNLIPromptBert.config \
    --gpu $gpus \



############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WikiREDPromptBert.config \
    --gpu $gpus \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptBert.config \
    --gpu $gpus \



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/STSBPromptBert.config \
    --gpu $gpus \
'''


'''
###
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/emobankarousalPromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/persuasivenessrelevancePromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/persuasivenessspecificityPromptBert.config \
    --gpu $gpus



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/emobankdominancePromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/squinkyimplicaturePromptBert.config \
    --gpu $gpus



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/squinkyformalityPromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptBert.config \
    --gpu $gpus

###
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/anliPromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastfactualityPromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptBert.config \
    --gpu $gpus

'''
