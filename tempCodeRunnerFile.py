#converting the dataframes to vector form
from sentence_transformers import SentenceTransformer
#using huggingface encoder
encoder=SentenceTransformer("all-mpnet-base-v2")
vectors=encoder.encode(df.text)
print(vectors)