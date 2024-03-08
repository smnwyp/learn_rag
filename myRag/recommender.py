'''
this script builds a recommendation system using Pinecone and OpenAI's API.
interesting thing about this task is how to compress long text, do we only compress and index the title?
or the content as well? which is better?

also, another interesting part is how to chunk long text, with overlap, to get the best embeddings?
'''
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm, trange
from util import Utils

import pandas as pd
import time
import os

utils = Utils()


def get_embeddings(articles: list[str], embedding_model: str, openai_client):
   return openai_client.embeddings.create(input=articles, model=embedding_model)


def preprocess_title_and_update_index(pinecone_index: Pinecone, embedding_model: str, file_path: str):
    CHUNK_SIZE = 400
    TOTAL_ROWS = 10000
    progress_bar = tqdm(total=TOTAL_ROWS)

    # read csv in chunks
    chunks = pd.read_csv(file_path, chunksize=CHUNK_SIZE, nrows=TOTAL_ROWS)
    chunk_num = 0
    for chunk in chunks:
        # vectorize all titles
        titles = chunk['title'].tolist()
        embeddings = get_embeddings(titles, embedding_model=embedding_model, openai_client=openai_client)
        # prepare batch for upsert
        prepped = [{'id': str(chunk_num * CHUNK_SIZE + i),
                    'values': embeddings.data[i].embedding,
                    'metadata': {'title': titles[i]}, } for i in range(0, len(titles))]
        chunk_num = chunk_num + 1
        # upsert in batch
        if len(prepped) >= 200:
            pinecone_index.upsert(prepped)
            prepped = []
        progress_bar.update(len(chunk))
    print(pinecone_index.describe_index_stats())


def get_recommendations(pinecone_index, search_term, top_k: int, embedding_model: str, openai_client: OpenAI):
  embed = get_embeddings([search_term], embedding_model=embedding_model,
                         openai_client=openai_client).data[0].embedding
  res = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
  return res


def upsert_embedding(embeddings, title: str, prepped, embed_num, pinecone_index: Pinecone, title_id: int):
   # embedding.data is list[embedding]
   for embedding in embeddings.data:
     prepped.append({'id': str(embed_num),
                    'values': embedding.embedding,
                    'metadata': {'title':title,
                                 'title_id': title_id}})
     embed_num += 1
     if len(prepped) >= 100:
         pinecone_index.upsert(prepped)
         prepped.clear()
   return embed_num


def preprocess_article_and_update_index(file_path: str, embedding_model: str,
                                        openai_client: OpenAI, pinecone_index: Pinecone):
    news_data_rows_num = 100

    embed_num = 0  # keep track of embedding number for 'id'

    # this is the interesting bit, how to chunk long text -- with overlap!
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                   chunk_overlap=20)  # how to chunk each article
    prepped = []
    df = pd.read_csv(file_path, nrows=news_data_rows_num)
    articles_list = df['article'].tolist()
    titles_list = df['title'].tolist()

    for i in range(0, len(articles_list)):
        # for every title, there might be multiple chunks of articles,
        # therefore, multiple item in the index may have the same `'metadata': {'title':title}}`
        art = articles_list[i]
        title = titles_list[i]
        if art is not None and isinstance(art, str):
            chunk_texts: list[str] = text_splitter.split_text(art)
            embeddings = get_embeddings(chunk_texts, embedding_model=embedding_model, openai_client=openai_client)
            embed_num = upsert_embedding(embeddings=embeddings, title=title, prepped=prepped,
                                         embed_num=embed_num, pinecone_index=pinecone_index, title_id=i)

    pinecone_index.describe_index_stats()


if __name__ == "__main__":
    embedding_model = "text-embedding-ada-002"
    file_path = '../data/all-the-news-3.csv'
    # load or create index
    pinecone_index = utils.create_or_get_pinecone_index(pinecone_index_name='recommender')
    openai_client = utils.get_openai_client()

    if_only_title = False
    if_not_loaded = False
    if if_not_loaded and if_only_title:
        # load, preprocess (vectorize) title only and update index
        preprocess_title_and_update_index(pinecone_index=pinecone_index, embedding_model=embedding_model,
                                          file_path=file_path)
    elif if_not_loaded and not if_only_title:
        # load, preprocess (vectorize) article and update index
        preprocess_article_and_update_index(file_path=file_path, embedding_model=embedding_model,
                           openai_client=openai_client, pinecone_index=pinecone_index)
    else:
        print('loading existing index.')

    # query for recommendations
    search_term = 'Kardashian'
    print(f'Querying for recommendations...{search_term=}')
    reco = get_recommendations(pinecone_index=pinecone_index, search_term=search_term,
                               top_k=10, embedding_model=embedding_model, openai_client=openai_client)
    for r in reco.matches:
        print(f'{r.id=}--{r.score=}: {r.metadata["title_id"]} -- {r.metadata["title"]}')