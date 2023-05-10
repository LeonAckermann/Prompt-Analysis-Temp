from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, T5TokenizerFast, T5ForConditionalGeneration

# Directories where you want to save the models
dir_bert = './BertForMaskedLM'
dir_t5 = './T5ForMaskedLM'
dir_t5_large = './T5LargeForMaskedLM'
dir_t5_small = './T5SmallForMaskedLM'
dir_roberta = './RobertaForMaskedLM'
dir_roberta_large = './RobertaLargeForMaskedLM'

# Initialize tokenizer and model
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer_t5_small = T5TokenizerFast.from_pretrained('t5-small')
model_t5_small = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer_t5 = T5TokenizerFast.from_pretrained('t5-base')
model_t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer_t5_large = T5TokenizerFast.from_pretrained('t5-large')
model_t5_large = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = RobertaForMaskedLM.from_pretrained('roberta-base')
tokenizer_roberta_large = RobertaTokenizer.from_pretrained('roberta-large')
model_roberta_large = RobertaForMaskedLM.from_pretrained('roberta-large')


# Save the tokenizer and model to the specified directory
tokenizer_bert.save_pretrained(dir_bert)
model_bert.save_pretrained(dir_bert)
tokenizer_t5_small.save_pretrained(dir_t5_small)
model_t5_small.save_pretrained(dir_t5_small)
tokenizer_t5.save_pretrained(dir_t5)
model_t5.save_pretrained(dir_t5)
tokenizer_t5_large.save_pretrained(dir_t5_large)
model_t5_large.save_pretrained(dir_t5_large)
tokenizer_roberta.save_pretrained(dir_roberta)
model_roberta.save_pretrained(dir_roberta)
tokenizer_roberta_large.save_pretrained(dir_roberta_large)
model_roberta_large.save_pretrained(dir_roberta_large)

