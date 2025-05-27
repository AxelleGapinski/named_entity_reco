import random

import spacy
from spacy.util import minibatch
from spacy.training.example import Example

train_data = [
    ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("How much are 20 chairs?", {"entities": [(13, 15, "QUANTITY"), (16, 22, "PRODUCT")]}),
    ("Give me the value of 5 laptops.", {"entities": [(21, 23, "QUANTITY"), (23, 30, "PRODUCT")]}),
    ("The price of 3 apples is $1.50.", {"entities": [(13, 14, "QUANTITY"), (15, 21, "PRODUCT")]}),
    ("I need to buy 15 oranges.", {"entities": [(14, 16, "QUANTITY"), (17, 24, "PRODUCT")]}),
    ("Can you tell me the cost of 8 books?", {"entities": [(28, 29, "QUANTITY"), (30, 35, "PRODUCT")]}),
    ("What is the price of 12 chairs?", {"entities": [(21, 23, "QUANTITY"), (24, 30, "PRODUCT")]}),
    ("I want to purchase 6 phones.", {"entities": [(19, 20, "QUANTITY"), (21, 28, "PRODUCT")]})
]

nlp = spacy.load("en_core_web_md")

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

for _, annotations in train_data:
    for ent in annotations.get("entities"):
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = spacy.blank('en')
    
    epochs = 50
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Epoch {epoch + 1}, Losses: {losses}")

nlp.to_disk("custom_ner_model")

trained_nlp = spacy.load("custom_ner_model")

test_texts = [
    "How much for 3 oranges?",
    "I want 15 chairs for the conference.",
    "Can you give me the price for 8 laptops?",
    "What is the price for 200 rugs?",
    "I need 10 phones for my team.",
    "The cost of 5 tablets is too high."
]

for text in test_texts:
    doc = trained_nlp(text)
    print(f"Text: {text}")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print()	