import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    

    # path to the provided dataset file
    data_path = os.path.join(os.path.dirname(__file__), "sift-128-euclidean.hdf5")

    # open the hdf5 and read train/test
    with h5py.File(data_path, 'r') as f:
        train = f['train'][:]
        test = f['test'][:]

    # FAISS expects float32
    xb = train.astype('float32')
    xq = test.astype('float32')

    d = xb.shape[1]

    # HNSW parameters
    M = 16
    ef_construction = 200
    ef_search = 200

    # Create HNSW index (IndexHNSWFlat uses inner product or L2 depending on metric; default is L2 for IndexHNSWFlat)
    # We'll use IndexHNSWFlat with L2 (faiss.METRIC_L2)
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction

    # add database vectors
    index.add(xb)

    # set ef for search
    index.hnsw.efSearch = ef_search

    # use the first query vector from test set
    q = xq[0:1]

    k = 10
    D, I = index.search(q, k)

    # I is shape (1, k) containing indices of nearest neighbours
    topk = I[0].tolist()

    out_path = os.path.join(os.path.dirname(__file__), 'output.txt')
    with open(out_path, 'w') as fo:
        for idx in topk:
            fo.write(str(int(idx)) + '\n')

    print(f"Wrote top {k} neighbor indices to {out_path}")

if __name__ == "__main__":
    evaluate_hnsw()
