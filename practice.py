import spacy
import pandas as pd

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Process the text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Create a list of dictionaries for each token
data = []
for token in doc:
    data.append({
        "Text": token.text,
        "Lemma": token.lemma_,
        "POS": token.pos_,
        "Tag": token.tag_,
        "Dep": token.dep_,
        "Shape": token.shape_,
        "Is Alpha": token.is_alpha,
        "Is Stop": token.is_stop
    })

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
