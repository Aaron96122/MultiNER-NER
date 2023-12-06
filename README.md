<div align="center">
<h1>
MultiNERD NER models fine-tuned on the English subset using Roberta and Spanmarker
</h1>

[MultiNERD Dataset](https://huggingface.co/datasets/Babelscape/multinerd?row=17) |
[roberta-base model](https://huggingface.co/roberta-base) |
[SpanMarker framework for NER](https://huggingface.co/docs/hub/span_marker) |
[Paper for Dataset](https://aclanthology.org/2022.findings-naacl.60.pdf) | 
</div>

MultiNERD NER is a Named Entity Recognition model specifically adjusted for the English subset within the MultiNERD dataset. The model comprises two systems, labeled A and B, and utilizes two frameworks, namely AutoModelForTokenClassification and SpanMarker. Each of these components undergoes fine-tuning based on the pre-trained language model of roberta-base. Further information about the two systems and corresponding code details are provided below. 
## Details:
### System A 
Fine-tune the roberta-base model on the English subset of the MultiNERD dataset.  
### System B 
Fine-tune the roberta-base model that will predict only five entity types and the O tag. All examples remain, but entity types not belonging to one of the following five are set to zero: PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM)
### Roberta
RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts.

### SpanMarker
SpanMarker is a framework for training powerful Named Entity Recognition models using familiar encoders such as BERT, RoBERTa and DeBERTa. Tightly implemented on top of the 🤗 Transformers library, SpanMarker can take good advantage of it. As a result, SpanMarker will be intuitive to use for anyone familiar with Transformers.

## Setting up the Docker environment and installing the dependencies
Please install the following dependencies:
```bash
pip install -r requirements.txt
```

### Run the code

The fine-tune RoBERTa for System A using AutoModelForTokenClassification:
```bash
python bert_a.py --model_name roberta-base \ """You can change to another pre-trained large language model."""
                --output_dir /home/yaruwu/rise/outputs \ """The output directory to store the final results of the test set."""
                --save_dir /home/yaruwu/rise/models \ """The final directory to save the fine-tuned model for further use.""" 
```
The fine-tune RoBERTa for System B:
```bash
python bert_b.py --model_name roberta-base \ """You can change to another pre-trained large language model."""
                --output_dir /home/yaruwu/rise/outputs \ """The output directory to store the final results of the test set."""
                --save_dir /home/yaruwu/rise/models \ """The final directory to save the fine-tuned model for further use.""" 
```
The fine-tune RoBERTa using SpanMarker for System A (Please remember to add the info regarding the wandb when running the models with the SpanMaker framework):
```bash
python spanmodel.py --system A \ """You can choose system A and B here.""" 
                --model_name roberta-base \ """You can change to another pre-trained large language model."""
                --output_dir /home/yaruwu/rise/outputs \ """The output directory to store the final results of the test set."""
                --save_dir /home/yaruwu/rise/models \ """The final directory to save the fine-tuned model for further use.""" 
```
The fine-tune RoBERTa using SpanMarker for System B:
```bash
python spanmodel.py --system A \ """You can choose system A and B here.""" 
                --model_name roberta-base \ """You can change to another pre-trained large language model."""
                --output_dir /home/yaruwu/rise/outputs \ """The output directory to store the final results of the test set."""
                --save_dir /home/yaruwu/rise/models \ """The final directory to save the fine-tuned model for further use."""
```
### Evaluation results

The overall performance of the two systems:

|              | Accuracy (entity)  | Recall (entity)    | Precision (entity)  | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |------------------ |
| RoBERTa-A | 0.9853     | 0.9684     | 0.8805    | 0.9223  |
| RoBERTa-B     | 0.9927 | 0.9800 | 0.9423 | 0.9608  |
| SpanMarker-RoBERTa-A    | 0.9896 | 0.9569 | 0.9380 | 0.9473   |
| SpanMarker-RoBERTa-B    | 0.9937 | 0.9749 | 0.9598 | **0.9673**   |
