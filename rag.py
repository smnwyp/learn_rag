from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from util import Utils

import ast
import os
import pandas as pd

def load_data(filename:str):
    max_articles_num = 500
    df = pd.read_csv(filename, nrows=max_articles_num)
    return df

def transform_upsert_embedding(df, pinecone_index: Pinecone):
    '''
    wikipedia data is already vectorized, so we just need to upsert it to Pinecone
    '''
    prepped = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        meta = ast.literal_eval(row['metadata'])
        prepped.append({'id': row['id'],
                        'values': ast.literal_eval(row['values']),
                        'metadata': meta})
        if len(prepped) >= 250:
            pinecone_index.upsert(prepped)
            prepped = []
    print(f'Upserted embeddings to Pinecone {pinecone_index.describe_index_stats()}')

def get_embeddings(articles, openai_client, model: str):
   return openai_client.embeddings.create(input=articles, model=model)

def run_semantic_query(query: str, pinecone_index: Pinecone, openai_client, model: str):
    embed = get_embeddings([query], openai_client=openai_client, model=model)
    res = pinecone_index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
    return res

def build_prompt(query_res, query):
    contexts = [x['metadata']['text'] for x in query_res['matches']]

    prompt_start = ("Answer the question based on the context below.\n\n" + "Context:\n")
    prompt_end = (f"\n\n Question: {query} \n Answer:")

    prompt = (prompt_start + "\n\n---\n\n".join(contexts) + prompt_end)

    system_msg = {"role": "system",
                  "content": "you are a helpful assistant that does whatever the user asks, within reason."}
    user_msg = {"role": "user", "content": prompt}
    msg = [system_msg, user_msg]

    return msg

def create_summary(openai_client, prompt: str, model: str):
    summary_res = openai_client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0,
        max_tokens=636,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    summary = summary_res.choices[0].message.content
    return summary

def do_augmented_query(query, pinecone_index, openai_client, model: str, llm: str):
    print(f"doing semantic search for query: {query}")
    query_res = run_semantic_query(query, pinecone_index, openai_client=openai_client, model=model)
    print(f"building augmented prompt for query")
    prompt_augmented_summary = build_prompt(query_res, query)
    print(f"generating summary using LLM: {llm} and given prompt")
    summary = create_summary(openai_client=openai_client, prompt=prompt_augmented_summary, model=llm)
    return summary

if __name__ == "__main__":
    utils = Utils()
    openai_client = utils.get_openai_client()
    pinecone_index = utils.create_or_get_pinecone_index('wiki-qa')
    embedding_model = "text-embedding-ada-002"
    llm = "gpt-4-0125-preview"

    not_loaded=False
    if not_loaded:
        df = load_data(filename='./data/wiki.csv')
        transform_upsert_embedding(df=df, pinecone_index=pinecone_index)

    query = "write a comprehensive summary in max 50 words about on: what is a scheibenbremse."
    summary = do_augmented_query(query=query, pinecone_index=pinecone_index, openai_client=openai_client,
                                 model=embedding_model, llm=llm)
    print(f"{summary=}")

