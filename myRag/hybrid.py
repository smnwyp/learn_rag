"""
this script is used to create a hybrid index using both dense and sparse vectors.
sparse vectors are created using BM25Encoder and dense vectors are created using SentenceTransformer.
The script also contains a function to run a query using the hybrid index, which uses a convex combination of dense and sparse vectors.
see https://docs.pinecone.io/docs/indexes#distance-metrics
one thing to note is the sparse encoder is 'trained' on the dataset, therefore to sparse-ly encode the query, the same encoder needs to be available.
and the consequence of using sparse encoder is the metrics of the index will be different from the dense index, it needs to be dotproduct.
The dense encoder is a pre-trained model, and the same model is used to encode the query and the images.
for hybrid search, see https://docs.pinecone.io/docs/query-sparse-dense-vectors
"""
from datasets import load_dataset
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
from util import Utils

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

def load_data(filename:str):
    fashion = load_dataset(filename, split="train")

    return fashion

def create_sparse_encoder(metadata):
    bm25 = BM25Encoder()
    bm25.fit(metadata['productDisplayName'])
    return bm25

def create_embedding_using_sparse_dense(input_data, sparse_encoder, dense_encoder, pinecone_index: Pinecone, sentence_transformer_path: str):
    batch_size = 100
    fashion_data_num = 1000
    images = input_data['image']

    metadata = input_data.remove_columns('image')
    metadata = metadata.to_pandas()

    for i in tqdm(range(0, min(fashion_data_num, len(input_data)), batch_size)):
        # find end of batch
        i_end = min(i + batch_size, len(input_data))
        # extract metadata batch
        meta_batch = metadata.iloc[i:i_end]
        meta_dict = meta_batch.to_dict(orient="records")
        # concatinate all metadata field except for id and year to form a single string
        meta_batch = [" ".join(x) for x in meta_batch.loc[:, ~meta_batch.columns.isin(['id', 'year'])].values.tolist()]
        # extract image batch
        img_batch = images[i:i_end]
        # create sparse BM25 vectors
        sparse_embeds = sparse_encoder.encode_documents([text for text in meta_batch])
        # create dense vectors
        dense_embeds = dense_encoder.encode(img_batch).tolist()
        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]

        upserts = []
        # loop through the data and create dictionaries for uploading documents to pinecone index
        for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
            upserts.append({
                'id': _id,
                'sparse_values': sparse,
                'values': dense,
                'metadata': meta
            })
        # upload the documents to the new hybrid index
        pinecone_index.upsert(upserts)

    # show index description after uploading the documents
    pinecone_index.describe_index_stats()

def hybrid_scale(dense, sparse, alpha: float):
    """Hybrid vector scaling using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: float between 0 and 1 where 0 == sparse only
               and 1 == dense only
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def query_hybrid(dense_encoder, sparse_encoder, pinecone_index, images, query: str, alpha:float=0.5):
    sparse = sparse_encoder.encode_queries(query)
    dense = dense_encoder.encode(query).tolist()
    # Closer to 0==more sparse, closer to 1==more dense
    hdense, hsparse = hybrid_scale(dense, sparse, alpha=alpha)
    result = pinecone_index.query(
        top_k=6,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )
    return result


if __name__ == "__main__":
    utils = Utils()
    pinecone_index = utils.create_or_get_pinecone_index('hybrid')

    input_data = "ashraq/fashion-product-images-small"
    device = utils.get_device()
    sentence_transformer_path = "sentence-transformers/clip-ViT-B-32"

    fashion = load_data(input_data)
    # create sparse encoder
    sparse_encoder = create_sparse_encoder(fashion)
    # create dense encoder
    dense_encoder = SentenceTransformer(sentence_transformer_path, device=device)

    if_need_to_load = False
    if if_need_to_load:
        print(f"Loading data")
        # create embedding using sparse and dense vectors and upload to pinecone index
        create_embedding_using_sparse_dense(input_data=fashion, pinecone_index=pinecone_index,
                                            sentence_transformer_path=sentence_transformer_path,
                                            sparse_encoder=sparse_encoder, dense_encoder=dense_encoder)
    else:
        print(f"Data already loaded, skipping loading data")


    print(f"Running query")
    query = "white jeans for woman"
    # alpha, closer to 0==more sparse, closer to 1==more dense
    result = query_hybrid(dense_encoder, sparse_encoder,
                          pinecone_index, fashion['image'], query=query, alpha=0.5)


    for x in result["matches"]:
        print(x["metadata"]['productDisplayName'])







