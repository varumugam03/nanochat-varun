import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat_varun.common import get_base_dir

BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # last datashard in the dataset
index_to_filename = lambda i: f"shard_{i:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1, max_parquets:int=None):
    #split can only be "train" or "val" - train is everything but the last parquet, val is only the last parquet
    #start = rank, step = world_size

    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1 if max_parquets is None else min(max_parquets, len(parquet_paths))] if split == "train" else parquet_paths[-1:]

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


def download_single_shard(index):
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filename} as it already exists")
        return True

    #make remote url
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    #download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(filepath + ".tmp", "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB cuz karpathy did so
                    if chunk:
                        f.write(chunk)
            os.rename(filepath + ".tmp", filepath)
            print(f"Downloaded {filename} successfully")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Failed to download {filename}: {e} (attempt {attempt}/{max_attempts})")

            #cleanup partials
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        pass
            
            #exponential backoff
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("--num-shards", type=int, default=-1, help="Number of shards to download")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers to use")
    args = parser.parse_args()

    #dataset download run (ran with this cmd) - python -m nanochat_varun.dataset --num-shards=101 --num-workers=8

    # num_shards = MAX_SHARD + 1 if args.num_shards == -1 else min(args.num_shards, MAX_SHARD + 1)
    # num_workers = args.num_workers

    # print(f"Downloading {num_shards} shards with {num_workers} workers")
    # print(f"Target Directory: {DATA_DIR}")
    # ids = list(range(num_shards))
    
    # with Pool(num_workers) as pool:
    #     results = pool.map(download_single_shard, ids)

    # successful = sum(1 for success in results if success)
    # print(f"Successfully downloaded {successful}/{num_shards} shards")

    #sanity check
    texts = list(parquets_iter_batched("train", max_parquets=4))
    print("Length of generator: ", len(texts))
    for i, text in enumerate(texts):
        print(f"page {i}: contains {len(text)} documents")
        for j, doc in enumerate(text):
            print(f"document {j}: {len(doc)} words")
            if j == 2:
                break
        if i == 1:
            break
