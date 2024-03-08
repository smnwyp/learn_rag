'''
This script demonstrates how to use Pinecone to index and search for semantically similar questions.
it uses the Quora dataset, we first embed the questions into vectors then upsert the embeddings to Pinecone.
when a query is recieved, it is first encoded with the same language model,
the embeddings are then used to find the most similar questions.
'''
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from util import Utils
from tqdm.auto import tqdm

utils = Utils()

def load_data(dataset_name:str):
    dataset = load_dataset(dataset_name, split='train[240000:290000]')
    questions = []
    for record in dataset['questions']:
        questions.extend(record['data'])
    question = list(set(questions))
    print(f'Number of questions: {len(questions)}')
    return question

def create_embedding(index, question):
    batch_size = 200
    vector_limit = 10000

    questions = question[:vector_limit]

    for i in tqdm(range(0, len(questions), batch_size)):
        # find end of batch
        i_end = min(i + batch_size, len(questions))
        # create IDs batch
        ids = [str(x) for x in range(i, i_end)]
        # create metadata batch
        metadatas = [{'data': text} for text in questions[i:i_end]]
        # create embeddings
        xc = model.encode(questions[i:i_end])
        # create records list for upsert
        records = zip(ids, xc, metadatas)
        # upsert to Pinecone
        index.upsert(vectors=records)
    index.describe_index_stats()

def run_query(query, index, model):
  embedding = model.encode(query).tolist()
  results = index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
  for result in results['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['data']}")

def index_docs(dataset_name: str, pinecone_index: str):
    # load quora questions
    questions = load_data(dataset_name=dataset_name)
    create_embedding(index=pinecone_index, question=questions)



if __name__ == "__main__":
    device = utils.get_device()

    # load model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f'Model loaded. {model}')

    # load or create index
    pinecone_index = utils.create_or_get_pinecone_index(pinecone_index_name='quoras')

    # index documents
    if_loaded = True
    if not if_loaded:
        print('loading index.')
        index_docs(dataset_name='quora', pinecone_index=pinecone_index)
    else:
        print('Index already loaded.')

    query = 'how many types of ev batteries are there?'
    print(f"Running query: {query}")
    run_query(query, index=pinecone_index, model=model)



