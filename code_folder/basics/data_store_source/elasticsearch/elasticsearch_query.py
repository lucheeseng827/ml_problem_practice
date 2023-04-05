from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost", "port": 9200}])


results = es.search(index="my_index", body={"query": {"match_all": {}}})
