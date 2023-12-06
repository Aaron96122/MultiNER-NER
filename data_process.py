import datasets
from datasets import load_dataset, Dataset, DatasetDict

ori_label2id = {
        "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
        "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8, "B-BIO": 9,
        "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, 
        "B-EVE": 15, "I-EVE": 16, "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, 
        "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23, "I-MYTH": 24, 
        "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, 
        "I-VEHI": 30
    }
ori_label = [key for key, value in ori_label2id.items()]

new_label2id = {
        "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
        "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8, "B-BIO": 0,
        "I-BIO": 0, "B-CEL": 0, "I-CEL": 0, "B-DIS": 9, "I-DIS": 10, 
        "B-EVE": 0, "I-EVE": 0, "B-FOOD": 0, "I-FOOD": 0, "B-INST": 0, 
        "I-INST": 0, "B-MEDIA": 0, "I-MEDIA": 0, "B-MYTH": 0, "I-MYTH": 0, 
        "B-PLANT": 0, "I-PLANT": 0, "B-TIME": 0, "I-TIME": 0, "B-VEHI": 0, 
        "I-VEHI": 0
    }
new_label = ['O'] + [key for key, value in new_label2id.items() if value != 0]

def transform(item):
    for key, value in ori_label2id.items():
      if value == item:
        return key
    
def trans(target_list):
    result = [new_label2id[transform(i)] for i in target_list]
    return result

def process_A(dataset):
    temp = dataset.to_pandas()
    eng_data = temp[temp['lang']=='en']
    result = Dataset.from_pandas(eng_data)
    label2id = ori_label2id
    id2label = {k:v for v, k in ori_label2id.items()}
    return result, label2id, id2label

def process_B(dataset):
    temp = dataset.to_pandas()
    eng_data = temp[temp['lang']=='en']
    eng_data['ner_tags'] = eng_data['ner_tags'].map(lambda x: trans(x))
    result = Dataset.from_pandas(eng_data)
    label2id = {v:k for k,v in enumerate(new_label)}
    id2label = {k:v for k,v in enumerate(new_label)}
    return result, label2id, id2label


  