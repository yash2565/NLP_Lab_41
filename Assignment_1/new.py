import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

print(doc)
for token in doc:
    print(token.text)
    
print("---------------------------------------------------------------------------")
import spacy

# English pipelines include a rule-based lemmatizer
nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.get_pipe("lemmatizer")
print(lemmatizer.mode)  # 'rule'

doc = nlp("I was reading the paper.")
print([token.lemma_ for token in doc])
# ['I', 'be', 'read', 'the', 'paper', '.']

print("---------------------------------------------------------------------------")

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:    
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

print("---------------------------------------------------------------------------")


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I live in New York")
print("Before:", [token.text for token in doc])

with doc.retokenize() as retokenizer:
    retokenizer.merge(doc[3:5], attrs={"LEMMA": "new york"})
print("After:", [token.text for token in doc])
