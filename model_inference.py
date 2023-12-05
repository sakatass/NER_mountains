import spacy
import json

def get_ner_json(text, ner_model_path):
    ner_model = spacy.load(ner_model_path)
    doc = ner_model(text)

    output_data = {"text": text, "ents": [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]}

    return output_data