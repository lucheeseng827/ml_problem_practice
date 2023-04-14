# This code searches for all documents in the index "my_index".
# It returns a list of all documents found.

from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost", "port": 9200}])

results = es.search(index="my_index", body={"query": {"match_all": {}}})
