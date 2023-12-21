import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext
)


import logging
import sys


def query_data():
    while True:
        question = input("Enter your question, or type 'exit' to quit: ")
        if question.lower() == 'exit':
            break
        response = index.as_query_engine().query(question)
        print(response)


def load_index(service_context):
    # check if storage already exists
    if not os.path.exists("./index"):
        # load the documents and create the index
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        # store it for later
        index.storage_context.persist(persist_dir='index')
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)


def main():
    # Load data and build an index
    service_context = ServiceContext.from_defaults(
    embed_model="local:BAAI/bge-large-en")
    load_index(service_context)
    
    # documents = SimpleDirectoryReader("data").load_data()
    # index = VectorStoreIndex.from_documents(documents)
    # enable logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    # querying your data
    query_data()

    # storing the index to disk instead of memory
    # index.storage_context.persist(persist_dir='index')


if __name__ == "__main__":
    main()
