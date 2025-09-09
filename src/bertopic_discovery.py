#!/usr/bin/env python3
"""
BERTopic discovery for short social posts (Russian/multilingual).

This script is compatible with both a *directory of CSV files* and a *single CSV file*.
Paths are configurable via CLI args or environment variables (loaded from .env if present).

Env/CLI variables:
- POSTS_PATH / POSTS_DIR: path to a folder with CSVs *or* a single CSV file.
- PARQUET_PATH: where to write the cleaned parquet file (default: data/all_posts.parquet).
- OUTPUTS_DIR: where to write model and CSV outputs (default: outputs/).
- MIN_DF (int or float): min_df for CountVectorizer (default: 0.01). If float in (0,1], treated as fraction.
- MAX_DF (int or float): max_df for CountVectorizer (default: 1.0). If float in (0,1], treated as fraction.
- MIN_CLUSTER_SIZE (int): HDBSCAN min_cluster_size (default: 50; lower for tiny test data).
- MIN_SAMPLES (int): HDBSCAN min_samples (default: 10).

Usage examples:
- Use .env defaults:
    python src/bertopic_discovery.py

- Override posts path with a single CSV for quick test:
    POSTS_PATH=../sns4human/data/vk/posts/religion/concerto.csv \
    python src/bertopic_discovery.py

- CLI args override env:
    python src/bertopic_discovery.py \
        --posts ../sns4human/data/vk/posts \
        --parquet data/all_posts.parquet \
        --outputs outputs
"""
import os
import re
import regex
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
from dotenv import load_dotenv


def normalize_text(s: str) -> str:
    """Minimal cleaning for short social texts while preserving meaning."""
    if not isinstance(s, str):
        return ""
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = re.sub(r"https?://\S+", " ", s)  # remove URLs
    s = regex.sub(r"#[\w\p{L}_-]+", " ", s, flags=regex.UNICODE)  # remove hashtags
    s = regex.sub(r"@[\w\p{L}_-]+", " ", s, flags=regex.UNICODE)  # remove mentions
    s = re.sub(r"\s+", " ", s)  # collapse spaces
    return s.strip()


def load_and_clean_posts(posts_path: str, output_path: str) -> pd.DataFrame:
    """Load CSV posts from given directory or file, clean them, and save parquet."""
    if os.path.isfile(posts_path):
        files = [posts_path]
    elif os.path.isdir(posts_path):
        files = sorted(glob(os.path.join(posts_path, '*.csv')))
    else:
        raise ValueError(f"Posts path not found: {posts_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    total_rows = 0
    for path in tqdm(files, desc='Loading CSVs'):
        try:
            df = pd.read_csv(path, encoding='utf-8')
            total_rows += len(df)
            keep = [c for c in ['id', 'text', 'date', 'group', 'likes', 'reposts', 'views'] if c in df.columns]
            df = df[keep].copy()
            df['source_file'] = os.path.basename(path)
            # Drop rows with missing or empty text
            df = df.dropna(subset=['text'])
            df = df[df['text'].astype(str).str.strip() != ""]
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Skip {path}: {e}")

    all_df = pd.concat(frames, ignore_index=True)
    all_df['text'] = all_df['text'].astype(str)
    all_df['text_clean'] = all_df['text'].map(normalize_text)
    all_df = all_df[all_df['text_clean'].str.len() >= 10].copy()
    all_df = all_df.drop_duplicates(subset=['text_clean']).reset_index(drop=True)

    cleaned_rows = len(all_df)
    if total_rows > 0:
        percent = cleaned_rows / total_rows * 100
        print(f"Total input rows: {total_rows}")
        print(f"Rows after cleaning: {cleaned_rows} ({percent:.2f}%)")
    else:
        print("Warning: No input rows found.")

    if 'date' in all_df.columns:
        try:
            all_df['date'] = pd.to_datetime(all_df['date'], errors='coerce')
        except Exception:
            pass

    all_df.to_parquet(output_path, index=False)
    print(f"Saved cleaned posts: {len(all_df)} rows -> {output_path}")
    return all_df


def _parse_df_param(env_name: str, n_docs: int, default: float):
    """Helper to parse MIN_DF / MAX_DF from env as float fraction."""
    val = os.environ.get(env_name)
    if val is None:
        return default
    try:
        if "." in val:
            return float(val)
        iv = int(val)
        return iv / n_docs if n_docs > 0 else default
    except Exception:
        return default


def build_bertopic_model(df: pd.DataFrame, outputs_dir: str):
    """Build BERTopic model and save outputs."""
    os.makedirs(outputs_dir, exist_ok=True)

    EMB_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    print(f"Loading embeddings model: {EMB_NAME}")
    embedder = SentenceTransformer(EMB_NAME)

    n_docs = len(df)
    print(f"Number of documents for BERTopic: {n_docs}")

    # Parse min_df and max_df from env or use defaults
    min_df = _parse_df_param('MIN_DF', n_docs, default=0.01)
    max_df = _parse_df_param('MAX_DF', n_docs, default=1.0)

    if max_df <= min_df:
        max_df = 1.0

    print(f"Vectorizer thresholds: min_df={min_df}, max_df={max_df}")

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None,
        min_df=min_df,
        max_df=max_df
    )

    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=5 if n_docs < 100 else 50,
        min_samples=2 if n_docs < 100 else 10,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="multilingual"
    )

    print("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(df['text_clean'].tolist())
    print("Model fitting completed.")

    model_path = os.path.join(outputs_dir, "bertopic_model")
    topic_model.save(model_path)
    print(f"Model saved to {model_path}")

    overview_path = os.path.join(outputs_dir, "topics_overview.csv")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(overview_path, index=False)
    print(f"Topics overview saved to {overview_path}")

    multi_path = os.path.join(outputs_dir, "post_topics_multi.csv")
    primary_path = os.path.join(outputs_dir, "post_topics_primary.csv")

    df_multi = pd.DataFrame({
        "post_id": df.index,
        "topic": topics,
        "probability": probs
    })
    df_multi.to_csv(multi_path, index=False)
    print(f"Per-post topic probabilities saved to {multi_path}")

    df_primary = df_multi.groupby("post_id").apply(lambda g: g.sort_values("probability", ascending=False).head(1))
    df_primary.to_csv(primary_path, index=False)
    print(f"Primary topic per post saved to {primary_path}")

    print(f"Outputs saved in {outputs_dir}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", type=str, help="Path to posts directory or single CSV file")
    args = parser.parse_args()

    posts_path = args.posts or os.environ.get("POSTS_PATH") or os.environ.get("POSTS_DIR", "data/raw_posts")
    parquet_path = os.environ.get("PARQUET_PATH", "data/all_posts.parquet")
    outputs_dir = os.environ.get("OUTPUTS_DIR", "outputs")

    df = load_and_clean_posts(posts_path, parquet_path)
    build_bertopic_model(df, outputs_dir)
