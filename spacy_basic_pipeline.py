import spacy

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a programming language that lets you work quickly.",
    "The Eiffel Tower is located in Paris, France.",
    "Elon Musk is the CEO of SpaceX and Tesla.",
    "Florian Dodov is the guy behind NeuralNine.",
    "What is the price of four bananas?",
    "How much are sixteen chairs?",
    "Give me the value of 5 laptops."
]

nlp = spacy.load("en_core_web_md")

# ner_labels = nlp.get_pipe("ner").labels
# print(ner_labels)

categories = ["ORG", "PERSON", "LOC"]
docs = [nlp(text) for text in texts]

for doc in docs:
    entities = []
    for ent in doc.ents:
        #if ent.label_ in categories:
        entities.append((ent.text, ent.label_))
    print(entities)