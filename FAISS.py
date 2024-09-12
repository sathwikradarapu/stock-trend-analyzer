# #Install Packages
# !pip install faiss-cpu
# !pip install sentence-transformers

# import pandas 
import pandas as pd 

df = pd.read_csv("FAISS_text.csv")



# Converting the text to vector using embeddings 

from sentence_transformers import SentenceTransformer

# Hugging face encoder 
encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(df.text)


dim = vectors.shape[1]



import faiss
index = faiss.IndexFlatL2(dim)




index.add(vectors)

search_query = "An apple a day keeps doctor away"

vec = encoder.encode(search_query)



import numpy as np 
svec = np.array(vec).reshape(1,-1)



index.search(svec, k=2)

dis, I = index.search(svec, k=2)



