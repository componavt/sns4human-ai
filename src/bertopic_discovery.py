
import os
import re
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan


def normalize_text(s: str) -> str:
    """Minimal cleaning for short social texts while preserving meaning."""
    if not isinstance(s, str):
        return ""
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = re.sub(r"https?://\S+", " ", s)  # remove URLs
    s = re.sub(r"#[\w\p{L}_-]+", " ", s, flags=re.UNICODE)  # remove hashtags
    s = re.sub(r"@[\w\p{L}_-]+", " ", s, flags=re.UNICODE)  # remove mentions
    s = re.sub(r"\s+", " ", s)  # collapse spaces
    return s.strip()


def load_and_clean_posts(posts_dir: str, output_path: str) -> pd.DataFrame:
    """Load CSV posts from given directory, clean them, and save parquet."""
    assert os.path.isdir(posts_dir), f"Posts directory not found: {posts_dir}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for path in tqdm(sorted(glob(os.path.join(posts_dir, '*.csv'))), desc='Loading CSVs'):
        try:
            df = pd.read_csv(path, encoding='utf-8')
            keep = [c for c in ['id', 'text', 'date', 'group', 'likes', 'reposts', 'views'] if c in df.columns]
            df = df[keep].copy()
            df['source_file'] = os.path.basename(path)
            frames.append(df)
        except Exception as e:
            print(f"Skip {path}: {e}")

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

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None,
        min_df=10,
        max_df=0.5
    )

    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
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

    # Save topics overview
    overview_path = os.path.join(outputs_dir, "topics_overview.csv")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(overview_path, index=False)

    # Save post-topic assignments
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

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # Configurable paths
    posts_dir = os.environ.get("POSTS_DIR", "data/raw_posts")  # set your path
    parquet_path = os.environ.get("PARQUET_PATH", "data/all_posts.parquet")
    outputs_dir = os.environ.get("OUTPUTS_DIR", "outputs")

    df = load_and_clean_posts(posts_dir, parquet_path)
    build_bertopic_model(df, outputs_dir)
