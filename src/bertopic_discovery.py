#!/usr/bin/env python3
"""
BERTopic discovery for short social posts (Russian/multilingual).

This script is compatible with both a *directory of CSV files* and a *single CSV file*.
Paths are configurable via CLI args or environment variables (loaded from .env if present).

Env/CLI variables:
- POSTS_PATH / POSTS_DIR: path to a folder with CSVs *or* a single CSV file.
- PARQUET_PATH: where to write the cleaned parquet file (default: data/all_posts.parquet).
- OUTPUTS_DIR: where to write model and CSV outputs (default: outputs/).
- MIN_DF (int or float): min_df for CountVectorizer (default: 10). If float in (0,1], treated as fraction.
- MAX_DF (int or float): max_df for CountVectorizer (default: 0.5). If float in (0,1], treated as fraction.
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
    for path in tqdm(files, desc='Loading CSVs'):
        try:
            df = pd.read_csv(path, encoding='utf-8')
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

    # Debug: print first 10 rows from frames after filtering non-empty text
    print("Debug: First 10 rows from frames after filtering non-empty text:")
    count = 0
    for frame in frames:
        # Print up to 10 rows in total from all frames
        for idx, row in frame.iterrows():
            print(row)
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break

    all_df = pd.concat(frames, ignore_index=True)
    all_df['text'] = all_df['text'].astype(str)
    all_df['text_clean'] = all_df['text'].map(normalize_text)
    all_df = all_df[all_df['text_clean'].str.len() >= 10].copy()
    all_df = all_df.drop_duplicates(subset=['text_clean']).reset_index(drop=True)

    if 'date' in all_df.columns:
        try:
            all_df['date'] = pd.to_datetime(all_df['date'], errors='coerce')
        except Exception:
            pass

    all_df.to_parquet(output_path, index=False)
    print(f"Saved cleaned posts: {len(all_df)} rows -> {output_path}")
    return all_df


def build_bertopic_model(df: pd.DataFrame, outputs_dir: str):
    """Build BERTopic model and save outputs."""
    os.makedirs(outputs_dir, exist_ok=True)

    EMB_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    embedder = SentenceTransformer(EMB_NAME)

    n_docs = len(df)
    min_df_value = 2 if n_docs < 100 else 10
    max_df_value = 0.5

    # --- FIX: ensure consistency of min_df and max_df ---
    if n_docs < min_df_value:
        min_df_value = 1

    if isinstance(max_df_value, float):
        max_df_abs = max_df_value * n_docs
    else:
        max_df_abs = max_df_value

    if max_df_abs < min_df_value:
        if n_docs > 1:
            max_df_value = max( (min_df_value + 1) / n_docs, min_df_value / n_docs )
        else:
            max_df_value = 1.0

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None,
        min_df=min_df_value,
        max_df=max_df_value
    )

    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=5 if len(df) < 100 else 50,
        min_samples=2 if len(df) < 100 else 10,
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

    topics, probs = topic_model.fit_transform(df['text_clean'].tolist())

    model_path = os.path.join(outputs_dir, "bertopic_model")
    topic_model.save(model_path)
    print(f"Model saved to {model_path}")

    overview_path = os.path.join(outputs_dir, "topics_overview.csv")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(overview_path, index=False)

    multi_path = os.path.join(outputs_dir, "post_topics_multi.csv")
    primary_path = os.path.join(outputs_dir, "post_topics_primary.csv")

    df_multi = pd.DataFrame({
        "post_id": df.index,
        "topic": topics,
        "probability": probs
    })
    df_multi.to_csv(multi_path, index=False)

    df_primary = df_multi.groupby("post_id").apply(lambda g: g.sort_values("probability", ascending=False).head(1))
    df_primary.to_csv(primary_path, index=False)

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
