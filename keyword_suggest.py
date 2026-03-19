"""
keyword_suggest.py
------------------
Hệ thống gợi ý từ khóa tìm kiếm sử dụng NLP (TF-IDF character n-grams)
kết hợp KNN (K-Nearest Neighbors) dựa trên dataset netflix_titles.csv.

Thuộc tính sử dụng: title, listed_in, description
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors


class KeywordSuggestor:
    """
    Hệ thống gợi ý từ khóa tìm kiếm sử dụng TF-IDF + KNN.

    Quy trình:
      1. Đọc dữ liệu từ netflix_titles.csv
      2. Xây dựng từ điển từ khóa từ 3 cột: title, listed_in, description
      3. Vector hóa từng từ bằng TF-IDF ký tự n-gram (character-level)
      4. Huấn luyện mô hình KNN với metric cosine
      5. Khi người dùng nhập, tìm k từ khóa gần nhất trong không gian vector
    """

    def __init__(self, data_path: str = "netflix_titles.csv"):
        self.data_path = data_path
        self.vocabulary: list[str] = []
        self.vocab_sources: dict[str, str] = {}   # word -> 'title' | 'genre' | 'keyword'

        # TF-IDF dùng character n-gram (2-4 ký tự) để bắt được prefix/infix
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=True,
            min_df=1,
        )

        # KNN với cosine distance (brute force vì vocab không quá lớn)
        self.knn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
        )

        self._is_trained = False

    # ------------------------------------------------------------------
    # Bước 1: Xây dựng từ điển từ khóa
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        """Chuẩn hóa cơ bản: bỏ ký tự đặc biệt thừa, strip whitespace."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _build_vocabulary(self, df: pd.DataFrame) -> list[str]:
        """
        Trích xuất tất cả từ khóa tiềm năng từ 3 cột:
          - title      → tiêu đề đầy đủ của từng bộ phim/chương trình
          - listed_in  → thể loại (genre)
          - description → từ đơn có nghĩa (loại stopword, giữ top-N)
        """
        vocab_set: set[str] = set()

        # --- Cột title ---
        if "title" in df.columns:
            for title in df["title"].dropna():
                t = self._normalize(str(title))
                if len(t) > 1:
                    vocab_set.add(t)
                    self.vocab_sources[t] = "title"

        # --- Cột listed_in (genre) ---
        if "listed_in" in df.columns:
            for genres in df["listed_in"].dropna():
                for genre in str(genres).split(","):
                    g = self._normalize(genre)
                    if len(g) > 1:
                        vocab_set.add(g)
                        self.vocab_sources.setdefault(g, "genre")

        # --- Cột description: trích top N từ bằng CountVectorizer ---
        if "description" in df.columns:
            try:
                cv = CountVectorizer(
                    stop_words="english",
                    max_features=3000,
                    lowercase=True,
                    token_pattern=r"[a-zA-Z]{3,}",  # chỉ lấy từ >= 3 ký tự
                )
                cv.fit(df["description"].dropna().astype(str))
                for word in cv.get_feature_names_out():
                    w = word.strip()
                    if w not in vocab_set:
                        vocab_set.add(w)
                        self.vocab_sources[w] = "keyword"
            except Exception as exc:
                print(f"[Warning] Không thể trích từ description: {exc}")

        return sorted(vocab_set)

    # ------------------------------------------------------------------
    # Bước 2: Huấn luyện mô hình
    # ------------------------------------------------------------------
    def train(self) -> bool:
        """
        Đọc CSV → xây dựng vocabulary → vector hóa TF-IDF → fit KNN.
        Trả về True nếu thành công.
        """
        print(f"[1/4] Đang tải dữ liệu từ '{self.data_path}'...")
        try:
            df = pd.read_csv(self.data_path, encoding="utf-8", on_bad_lines="skip")
            print(f"      Đã tải {len(df):,} bản ghi.")
        except Exception as exc:
            print(f"[Lỗi] Không thể đọc file: {exc}")
            return False

        print("[2/4] Đang xây dựng từ điển từ khóa (title + listed_in + description)...")
        self.vocabulary = self._build_vocabulary(df)
        print(f"      Tổng số từ khóa: {len(self.vocabulary):,}")

        print("[3/4] Vector hóa vocabulary bằng TF-IDF character n-gram...")
        X = self.vectorizer.fit_transform(self.vocabulary)
        print(f"      Ma trận vector: {X.shape[0]} từ × {X.shape[1]} features")

        print("[4/4] Huấn luyện mô hình KNN (cosine metric)...")
        self.knn.fit(X)
        self._is_trained = True
        print("      Huấn luyện hoàn tất!\n")
        return True

    # ------------------------------------------------------------------
    # Bước 3: Gợi ý từ khóa
    # ------------------------------------------------------------------
    def suggest(self, query: str, top_k: int = 8) -> list[dict]:
        """
        Gợi ý top_k từ khóa gần nhất với query.

        Trả về danh sách dict:
          {"keyword": str, "score": float, "type": str}
        """
        if not self._is_trained or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query.lower()])
        n_neighbors = min(top_k, len(self.vocabulary))
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=n_neighbors)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = round(float(1.0 - dist), 4)
            kw = self.vocabulary[idx]
            results.append({
                "keyword": kw,
                "score": similarity,
                "type": self.vocab_sources.get(kw, "keyword"),
            })

        # Sắp xếp theo điểm tương đồng giảm dần
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ------------------------------------------------------------------
# Demo CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "netflix_titles.csv")

    suggestor = KeywordSuggestor(csv_path)
    if not suggestor.train():
        exit(1)

    print("=== Hệ thống gợi ý từ khóa tìm kiếm (NLP + KNN) ===")
    print("Nhập từ khóa để nhận gợi ý (gõ 'quit' để thoát):\n")

    test_queries = ["strang", "com", "love", "doc", "thril", "sci", "kid", "murd", "rom", "acti"]
    for q in test_queries:
        suggestions = suggestor.suggest(q, top_k=5)
        names = [f"{s['keyword']} ({s['type']}, {s['score']:.3f})" for s in suggestions]
        print(f"Nhập: '{q:8s}' → {names}")

    print()
    while True:
        try:
            user_input = input("Nhập từ: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input:
                results = suggestor.suggest(user_input, top_k=8)
                print("Gợi ý:")
                for r in results:
                    print(f"  [{r['type']:8s}] {r['keyword']:<40s}  sim={r['score']:.4f}")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nKết thúc chương trình.")
