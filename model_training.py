import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en")
db = DocBin()

import json
f = open('/content/drive/MyDrive/Custom_NER_Mountains/train_data/train.json')
TRAIN_DATA = json.load(f)

for text, annot in tqdm(TRAIN_DATA['annotations']):
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("/content/drive/MyDrive/Custom_NER_Mountains/train_data/training_data.spacy")

import json
f = open('/content/drive/MyDrive/Custom_NER_Mountains/test_data/test.json')
TEST_DATA = json.load(f)

for text, annot in tqdm(TEST_DATA['annotations']):
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("/content/drive/MyDrive/Custom_NER_Mountains/test_data/testing_data.spacy")

!python -m spacy init fill-config base_config.cfg config.cfg

!python -m spacy train --gpu-id 0 config.cfg --output ./ --paths.train /content/drive/MyDrive/Custom_NER_Mountains/train_data/training_data.spacy --paths.dev /content/drive/MyDrive/Custom_NER_Mountains/test_data/testing_data.spacy

