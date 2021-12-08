import re
import string

PUNCUATION_LIST = list(string.punctuation)

def remove_punctuation(word_list):
    """Remove punctuation tokens from a list of tokens"""
    return ''.join([w for w in word_list if w not in PUNCUATION_LIST])


def preprocess_text(line):
    if line[:3] == 'RT ':
        line = line[3:]
    cleaned_line = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', line)
    cleaned_line = re.sub(r'pic\.twitter\.com\/\w+',' ',cleaned_line, re.UNICODE)
    cleaned_line = re.sub(r'(@)(\w+)',' ',cleaned_line)
    cleaned_line = re.sub(r'(#)(\w+)','\g<2>',cleaned_line)
    cleaned_line = re.sub(r'\n',' ',cleaned_line)
    cleaned_line = re.sub(r'(_|\…|#|@)',' ',cleaned_line)
    cleaned_line = deEmojify(cleaned_line)
    cleaned_line = re.sub(r'\.+',' . ',cleaned_line)
    # cleaned_line = re.sub(r'(â|œ|Œ|Â|ã|ƒ|Â|ð|Ÿ)', ' ', cleaned_line)
    cleaned_line = cleaned_line.lower()
    cleaned_line = cleaned_line.strip()
    cleaned_line = detect_elongated_words(cleaned_line)
    cleaned_line = remove_punctuation(cleaned_line)
    cleaned_line = re.sub(r'\s+',' ',cleaned_line)
    return cleaned_line

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def replace_elongated_word(word):
    regex = r'(\w*)([^\W\d_])(\2{2,})(\w*)'
    repl = r'\1\2\2\4'
    new_word = re.sub(regex, repl, word, flags=re.UNICODE)
    if new_word != word:
        return replace_elongated_word(new_word)
    else:
        return new_word

def detect_elongated_words(row):
    regexrep = r'(\w*)([^\W\d_])(\2{2,})(\w*)'
    words = [''.join(i) for i in re.findall(regexrep, row, flags=re.UNICODE)]
    for word in words:
        row = row.replace(word, replace_elongated_word(word))
    return row


from pathlib import Path

dataset_name = 'hasoc-2020/task2-ta'
val_set = False

dataset_path = Path('../datasets/offensive_2020_csv/') / dataset_name

import csv
from sklearn.model_selection import train_test_split

labels_to_val = {'not_offensive': 0, 'offensive': 1, 'not': 0, 'off': 1}

def read_offensive_split(filename, headers=False):
    if str(filename).endswith('.csv'):
        sep = ','
    elif str(filename).endswith('.tsv'):
        sep = '\t'
    else:
        raise Exception("Not CSV/TSV")

    texts = []
    labels = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=sep)
        if headers:
          header = next(reader)
        for row in reader:
            if len(row) == 2:
                text = row[0]
                label = row[1].strip()
            else:
                text = row[1]
                label = row[2].strip()

            texts.append(text)
            label = label.lower()
            labels.append(labels_to_val[label])

    return texts, labels

train_texts, train_labels = read_offensive_split( dataset_path / 'train.csv', headers=True)

if val_set:
  val_texts, val_labels = read_offensive_split( dataset_path / 'val.tsv')
else:
  train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, random_state=42, test_size=0.15, stratify=train_labels)

# Few shot training
train_size = 0.1
train_texts, _, train_labels, _ = train_test_split(train_texts, train_labels, random_state=42, train_size=train_size, stratify=train_labels)

print(f"Number of train samples : {len(train_labels)}")

train_text_processed = [preprocess_text(text) for text in train_texts]

val_text_processed = [preprocess_text(text) for text in val_texts]

from datasets import Dataset

train_dataset = Dataset.from_dict({'text': train_text_processed, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_text_processed, 'label': val_labels})

train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_encodings = train_dataset.map(tokenize_function, batched=True)
val_encodings = val_dataset.map(tokenize_function, batched=True)

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from transformers import set_seed

from sklearn.utils import class_weight
import torch.nn as nn
import torch

# Set seed value
set_seed(42)

device = 'cuda'
loss_weights = class_weight.compute_class_weight('balanced', np.unique(train_dataset['label']), train_dataset['label'])
loss_weights = np.exp(loss_weights)/np.sum(np.exp(loss_weights))
class_weights = torch.FloatTensor(loss_weights).to(device)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


num_epochs = 15
warmup_steps = 250
training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    learning_rate=2e-5,
    #evaluation_strategy='epoch',
    output_dir='./models',          # output directory
    num_train_epochs=num_epochs,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    save_steps=10000,
    save_total_limit=2,
    warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    #eval_steps=107,
    eval_steps=45,
    evaluate_during_training=True,
    logging_steps=9,
)

model_name = "xlm-roberta-base"
#model = AutoModelForSequenceClassification.from_pretrained(model_name)
#model.train()

# Load meta bert model

from meta_bert import MetaBERT, MetaBERTForHF
is_distil = False
is_xlm = True
bert = AutoModelForSequenceClassification.from_pretrained(model_name)
bert.eval()
t = bert.state_dict()
config = bert.config
model = MetaBERTForHF.init_from_pretrained(
    t,
    config,
    num_labels=2,
    is_distil=is_distil,
    is_xlm=is_xlm,
    per_step_layer_norm_weights=False,
    num_inner_loop_steps=5,
)

saved_models_filepath = "/home/azureuser/meta/ml_code/offensive_lang_detect_binary_proto/saved_models/"
checkpoint = Path(saved_models_filepath) / "train_model_best"

if checkpoint.exists():
    #Load the model
    print("Loading model")
    state = model.load_model(
        model_save_dir=saved_models_filepath,
        model_name="train_model",
        model_idx="best",
    )
    del state


trainer = WeightedTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_encodings,         # training dataset
    eval_dataset=val_encodings,             # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()


val_raw_pred,_,_ = trainer.predict(val_encodings)

import numpy as np
val_preds = np.argmax(val_raw_pred, axis=1)

val_encoded_labels = np.array(val_dataset['label'])

from sklearn.metrics import classification_report

report = classification_report(val_encoded_labels, val_preds, target_names=['NOT', 'OFF'])
print(report)

test_texts, test_labels = read_offensive_split(dataset_path / 'test.tsv')

test_text_processed = [preprocess_text(text) for text in test_texts]

test_dataset = Dataset.from_dict({'text': test_text_processed, 'label': test_labels})

test_encodings = test_dataset.map(tokenize_function, batched=True)

test_raw_pred,_,_ = trainer.predict(test_encodings)
test_preds = np.argmax(test_raw_pred, axis=1)

test_encoded_labels = np.array(test_dataset['label'])

report = classification_report(test_encoded_labels, test_preds, target_names=['NOT', 'OFF'])
print(report)
