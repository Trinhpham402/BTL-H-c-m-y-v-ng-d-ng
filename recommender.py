"""
recommender.py
--------------
Hệ thống gợi ý nội dung (Content-Based Filtering) cho Netflix dataset.
Sử dụng TF-IDF word-level trên tổ hợp title + listed_in + description
kết hợp K-Nearest Neighbors để tìm các bộ phim tương tự.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class NetflixRecommender:
    """
    Gợi ý nội dung Netflix dựa trên nội dung (Content-Based Filtering).

    Quy trình:
      1. Đọc CSV và tiền xử lý
      2. Ghép 3 cột (title, listed_in, description) thành "combined_text"
      3. Vector hóa bằng TF-IDF word-level
      4. Huấn luyện KNN
      5. Tìm k bộ phim tương tự nhất khi người dùng chọn 1 tiêu đề
    """

    def __init__(self, data_path: str = "netflix_titles.csv"):
        self.data_path = data_path
        self.df: pd.DataFrame | None = None

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 2),  # unigrams + bigrams
            sublinear_tf=True,
        )
        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._is_trained = False

    # ------------------------------------------------------------------
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Điền giá trị thiếu và tạo cột combined_text."""
        df = df.copy()
        for col in ["title", "listed_in", "description"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
            else:
                df[col] = ""

        # Tạo văn bản gộp: title được nhân đôi để tăng trọng số
        df["combined_text"] = (
            df["title"] + " " + df["title"] + " " +
            df["listed_in"].str.replace(",", " ") + " " +
            df["description"]
        )
        return df

    # ------------------------------------------------------------------
    def train(self) -> bool:
        """Tải data và huấn luyện mô hình KNN."""
        print(f"[1/3] Đang tải dữ liệu từ '{self.data_path}'...")
        try:
            df = pd.read_csv(self.data_path, encoding="utf-8", on_bad_lines="skip")
            print(f"      Đã tải {len(df):,} bản ghi.")
        except Exception as exc:
            print(f"[Lỗi] {exc}")
            return False

        print("[2/3] Tiền xử lý và vector hóa TF-IDF...")
        self.df = self._preprocess(df).reset_index(drop=True)
        X = self.vectorizer.fit_transform(self.df["combined_text"])
        print(f"      Ma trận: {X.shape[0]} bộ phim × {X.shape[1]} features")

        print("[3/3] Huấn luyện KNN...")
        self.knn.fit(X)
        self._is_trained = True
        print("      Huấn luyện hoàn tất!\n")
        return True

    # ------------------------------------------------------------------
    def recommend(self, title: str, top_k: int = 6) -> list[dict]:
        """
        Tìm top_k bộ phim tương tự với `title`.
        Trả về danh sách dict chứa thông tin của mỗi bộ phim.
        """
        if not self._is_trained:
            return []

        # Tìm kiếm linh hoạt (case-insensitive, partial match)
        mask = self.df["title"].str.lower().str.contains(title.lower(), na=False)
        matches = self.df[mask]

        if matches.empty:
            return []

        # Lấy bộ phim đầu tiên khớp
        idx = matches.index[0]
        query_vec = self.vectorizer.transform([self.df.loc[idx, "combined_text"]])
        n_neighbors = min(top_k + 1, len(self.df))
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=n_neighbors)

        results = []
        for dist, nbr_idx in zip(distances[0], indices[0]):
            if nbr_idx == idx:
                continue  # bỏ qua chính nó
            row = self.df.iloc[nbr_idx]
            results.append({
                "title": row.get("title", ""),
                "type": row.get("type", ""),
                "listed_in": row.get("listed_in", ""),
                "description": row.get("description", ""),
                "release_year": str(row.get("release_year", "")),
                "rating": row.get("rating", ""),
                "similarity": round(float(1.0 - dist), 4),
            })
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        """Tìm kiếm tiêu đề chứa query (hỗ trợ autocomplete)."""
        if self.df is None:
            return []
        mask = self.df["title"].str.lower().str.contains(query.lower(), na=False)
        return self.df[mask]["title"].head(limit).tolist()


# ------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "netflix_titles.csv")
    rec = NetflixRecommender(csv_path)
    if rec.train():
        for test_title in ["Stranger Things", "Bird Box", "The Crown"]:
            print(f"\nGợi ý tương tự '{test_title}':")
            for r in rec.recommend(test_title, top_k=4):
                print(f"  {r['title']:<40s}  sim={r['similarity']:.4f}  [{r['listed_in']}]")
