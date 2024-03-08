import os
import sys
from dotenv import load_dotenv, find_dotenv

from pinecone import Pinecone
import torch
from openai import OpenAI


class Utils:
    def __init__(self):
        pass

    def get_openai_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("OPENAI_API_KEY")

    def get_pinecone_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("PINECONE_API_KEY")

    def get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print('Sorry no cuda.')

    def create_or_get_pinecone_index(self, pinecone_index_name: str):
        PINECONE_API_KEY = self.get_pinecone_api_key()
        pinecone = Pinecone(api_key=PINECONE_API_KEY)

        index = pinecone.Index(pinecone_index_name)
        print(f'Index {pinecone_index_name} created.')
        return index

    def get_openai_client(self):
        OPENAI_API_KEY = self.get_openai_api_key()
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return openai_client