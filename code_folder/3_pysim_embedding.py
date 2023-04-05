from gensim.models import Word2Vec

documents = [result["_source"]["field_name"] for result in results["hits"]["hits"]]

model = Word2Vec(documents, size=100, window=5, min_count=1)


# after embedding, you can store into data store  like mysql or panda

"""Store the embeddings in a secondary data store, such as a database or file system. You can use libraries like pandas or PyMySQL to write the embeddings to a database, or pickle to save them to a file."""
