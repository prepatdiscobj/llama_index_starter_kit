import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext
)

import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

import logging
import sys

LOGGING = False


def create_hugginface_llm(model_name="stabilityai/stablelm-zephyr-3b", 
                         tokenizer_name="stabilityai/stablelm-zephyr-3b",
                         prompt_template=PromptTemplate(
            "<|system|>\n<|endoftext|>\n<|user|>\n{query_str}<|endoftext|>\n<|assistant|>\n")):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    def messages_to_prompt(messages):

        prompt = ""
        for message in messages:
            if message.role == 'system':
                prompt += f"<|system|>\n{message.content}<|endoftext|>\n"
            elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}<|endoftext|>\n"
            elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}<|endoftext|>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n<|endoftext|>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"

        return prompt

    llm = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        query_wrapper_prompt=prompt_template,
        context_window=3900,
        max_new_tokens=256,
        model_kwargs={"quantization_config": quantization_config,
                      'trust_remote_code': True},  # this is to allow remote execution of model
        # tokenizer_kwargs={},
        generate_kwargs={"temperature": 0.8},
        messages_to_prompt=messages_to_prompt,
        device_map="auto",
    )

    return llm


def query_data(index):
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
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context)
        # store it for later
        index.storage_context.persist(persist_dir='index')
    else:
        # load the existing index
        print('Founding Already built-in Index. Loading from it')
        storage_context = StorageContext.from_defaults(persist_dir="./index")
        index = load_index_from_storage(storage_context)


def main():
    # setup the LLM
    print('Creating Huggingface LLM.....', end='')
    llm = create_hugginface_llm()
    print('Done\nCreating Index......',end='')
    service_context = ServiceContext.from_defaults(llm=llm,
        embed_model="local:BAAI/bge-small-en-v1.5")
    index = load_index(service_context)
    print('Done\n')
    # Load data and build an index  
    
    # enable logging
    if LOGGING:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    # querying your data
    query_data(index)


if __name__ == "__main__":
    main()
