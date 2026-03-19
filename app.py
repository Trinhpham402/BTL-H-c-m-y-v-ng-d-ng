"""
app.py
------
Flask web app: Hệ thống gợi ý từ khóa tìm kiếm NLP + KNN
Tích hợp KeywordSuggestor (gợi ý khi gõ) và NetflixRecommender (gợi ý nội dung tương tự)
"""

import os
from flask import Flask, render_template_string, request, jsonify
from keyword_suggest import KeywordSuggestor
from recommender import NetflixRecommender

# ── Khởi động Flask ─────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Tải và huấn luyện mô hình một lần khi khởi động ────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "netflix_titles.csv")

print("=" * 60)
print("  Khởi tạo Keyword Suggestor (NLP + KNN)...")
print("=" * 60)
suggestor = KeywordSuggestor(CSV_PATH)
suggestor.train()

print("=" * 60)
print("  Khởi tạo Netflix Recommender (Content-Based KNN)...")
print("=" * 60)
recommender = NetflixRecommender(CSV_PATH)
recommender.train()

# ── HTML Template ────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Netflix – Hệ thống gợi ý từ khóa NLP+KNN</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d0d0d;
      --surface:  #141414;
      --card:     #1e1e1e;
      --border:   #2a2a2a;
      --red:      #e50914;
      --red-glow: rgba(229,9,20,.35);
      --text:     #f5f5f5;
      --muted:    #8c8c8c;
      --badge-title:  #e50914;
      --badge-genre:  #3b82f6;
      --badge-kw:     #10b981;
      --radius:   12px;
      --transition: .22s cubic-bezier(.4,0,.2,1);
    }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: linear-gradient(180deg, #000 0%, transparent 100%);
      padding: 24px 48px;
      display: flex;
      align-items: center;
      gap: 16px;
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 100;
      backdrop-filter: blur(12px);
    }
    .logo {
      font-size: 1.9rem;
      font-weight: 700;
      color: var(--red);
      letter-spacing: -1px;
    }
    .tagline {
      font-size: .78rem;
      color: var(--muted);
      font-weight: 400;
      border-left: 2px solid var(--border);
      padding-left: 14px;
      margin-left: 4px;
    }

    /* ── Hero ── */
    .hero {
      text-align: center;
      padding: 72px 24px 40px;
      background: radial-gradient(ellipse 70% 40% at 50% 0%, rgba(229,9,20,.15) 0%, transparent 70%);
    }
    .hero h1 {
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 700;
      line-height: 1.2;
      margin-bottom: 12px;
    }
    .hero h1 span { color: var(--red); }
    .hero p {
      color: var(--muted);
      font-size: .95rem;
      max-width: 540px;
      margin: 0 auto 40px;
      line-height: 1.6;
    }

    /* ── Search Box ── */
    .search-wrap {
      position: relative;
      max-width: 680px;
      margin: 0 auto;
    }
    .search-icon {
      position: absolute;
      left: 20px; top: 50%; transform: translateY(-50%);
      color: var(--muted);
      font-size: 1.1rem;
      pointer-events: none;
    }
    #searchInput {
      width: 100%;
      padding: 18px 22px 18px 52px;
      font-size: 1.05rem;
      font-family: inherit;
      background: var(--card);
      border: 2px solid var(--border);
      border-radius: 999px;
      color: var(--text);
      outline: none;
      transition: border-color var(--transition), box-shadow var(--transition);
    }
    #searchInput:focus {
      border-color: var(--red);
      box-shadow: 0 0 0 4px var(--red-glow);
    }
    #searchInput::placeholder { color: var(--muted); }

    /* ── Dropdown ── */
    #dropdown {
      position: absolute;
      top: calc(100% + 8px);
      left: 0; right: 0;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 20px 60px rgba(0,0,0,.6);
      display: none;
      z-index: 200;
    }
    .dd-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 20px;
      cursor: pointer;
      transition: background var(--transition);
      border-bottom: 1px solid var(--border);
    }
    .dd-item:last-child { border-bottom: none; }
    .dd-item:hover { background: rgba(255,255,255,.05); }
    .dd-item-text { flex: 1; font-size: .92rem; }
    .dd-item-score {
      font-size: .75rem;
      color: var(--muted);
      font-family: monospace;
    }
    .badge {
      font-size: .65rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: .5px;
      flex-shrink: 0;
    }
    .badge-title  { background: rgba(229,9,20,.2);  color: var(--badge-title); }
    .badge-genre  { background: rgba(59,130,246,.2); color: var(--badge-genre); }
    .badge-keyword{ background: rgba(16,185,129,.2); color: var(--badge-kw); }
    .dd-empty {
      padding: 20px;
      text-align: center;
      color: var(--muted);
      font-size: .88rem;
    }

    /* ── Tabs ── */
    .main-content { max-width: 960px; margin: 0 auto; padding: 48px 24px; }
    .tabs {
      display: flex;
      gap: 4px;
      margin-bottom: 32px;
      background: var(--card);
      border-radius: 999px;
      padding: 4px;
      width: fit-content;
    }
    .tab-btn {
      padding: 9px 22px;
      background: none;
      border: none;
      border-radius: 999px;
      color: var(--muted);
      font-family: inherit;
      font-size: .88rem;
      font-weight: 500;
      cursor: pointer;
      transition: all var(--transition);
    }
    .tab-btn.active {
      background: var(--red);
      color: #fff;
      box-shadow: 0 4px 16px var(--red-glow);
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* ── Section Cards ── */
    .section-title {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 20px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      transition: transform var(--transition), border-color var(--transition), box-shadow var(--transition);
      cursor: pointer;
    }
    .card:hover {
      transform: translateY(-4px);
      border-color: var(--red);
      box-shadow: 0 12px 32px rgba(0,0,0,.4);
    }
    .card-type {
      font-size: .7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--red);
      margin-bottom: 8px;
    }
    .card-title {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 6px;
      line-height: 1.3;
    }
    .card-meta {
      font-size: .78rem;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .card-desc {
      font-size: .8rem;
      color: #bbb;
      line-height: 1.5;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .card-sim {
      display: inline-block;
      margin-top: 12px;
      font-size: .72rem;
      background: rgba(229,9,20,.15);
      color: var(--red);
      border-radius: 999px;
      padding: 3px 10px;
    }

    /* ── Stats Panel ── */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      text-align: center;
    }
    .stat-value {
      font-size: 2rem;
      font-weight: 700;
      color: var(--red);
    }
    .stat-label {
      font-size: .8rem;
      color: var(--muted);
      margin-top: 4px;
    }

    /* ── Loading ── */
    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--red);
      border-radius: 50%;
      animation: spin .7s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Instructions ── */
    .instructions {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px;
      margin-bottom: 32px;
    }
    .instructions h3 { margin-bottom: 12px; font-size: .95rem; }
    .instructions ol { padding-left: 20px; }
    .instructions li { font-size: .85rem; color: #bbb; margin-bottom: 8px; line-height: 1.5; }
    .instructions code {
      background: rgba(229,9,20,.15);
      color: var(--red);
      padding: 1px 6px;
      border-radius: 4px;
      font-size: .82rem;
    }

    /* ── Responsive ── */
    @media (max-width: 600px) {
      header { padding: 16px 20px; }
      .hero { padding: 48px 16px 32px; }
      .main-content { padding: 32px 16px; }
    }
  </style>
</head>
<body>

<header>
  <div class="logo">NETFLIX</div>
  <div class="tagline">Hệ thống gợi ý từ khóa tìm kiếm · NLP + KNN</div>
</header>

<section class="hero">
  <h1>Tìm kiếm thông minh<br>bằng <span>NLP & KNN</span></h1>
  <p>Nhập một vài ký tự để nhận gợi ý từ khóa tức thì từ hơn 8.000 tiêu đề Netflix, thể loại và từ khóa mô tả.</p>
  <div class="search-wrap">
    <span class="search-icon">🔍</span>
    <input
      id="searchInput"
      type="text"
      placeholder="Ví dụ: strang, love, doc, thril…"
      autocomplete="off"
      spellcheck="false"
    />
    <div id="dropdown"></div>
  </div>
</section>

<main class="main-content">
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('suggest')">💡 Gợi ý từ khóa</button>
    <button class="tab-btn" onclick="switchTab('recommend')">🎬 Gợi ý nội dung</button>
    <button class="tab-btn" onclick="switchTab('about')">📊 Thông tin mô hình</button>
  </div>

  <!-- Tab 1: Keyword suggestions -->
  <div id="tab-suggest" class="tab-panel active">
    <div class="instructions">
      <h3>🛠 Cách hoạt động</h3>
      <ol>
        <li>Nhập từ khóa vào ô tìm kiếm phía trên (ít nhất <code>2 ký tự</code>).</li>
        <li>Hệ thống vector hóa query bằng <code>TF-IDF character n-gram (2-4 ký tự)</code>.</li>
        <li>Mô hình <code>KNN (cosine metric)</code> tìm k từ khóa gần nhất trong từ điển.</li>
        <li>Kết quả phân loại: <code>title</code> (tiêu đề), <code>genre</code> (thể loại), <code>keyword</code> (từ khóa từ mô tả).</li>
      </ol>
    </div>
    <p class="section-title">Kết quả gợi ý xuất hiện tại dropdown khi bạn gõ ↑</p>
    <div id="suggest-results" class="cards-grid"></div>
  </div>

  <!-- Tab 2: Content recommendations -->
  <div id="tab-recommend" class="tab-panel">
    <div class="instructions">
      <h3>🎬 Gợi ý nội dung tương tự</h3>
      <ol>
        <li>Nhập tên bộ phim vào ô tìm kiếm và nhấn <code>Enter</code> hoặc chọn từ gợi ý.</li>
        <li>Hệ thống tìm bộ phim khớp rồi vector hóa bằng <code>TF-IDF word-level + bigram</code>.</li>
        <li>Mô hình <code>KNN</code> tìm các bộ phim có nội dung tương tự nhất.</li>
      </ol>
    </div>
    <p class="section-title">Gợi ý nội dung tương tự</p>
    <div id="recommend-results" class="cards-grid"></div>
  </div>

  <!-- Tab 3: About -->
  <div id="tab-about" class="tab-panel">
    <div class="stats-grid" id="stats-grid">
      <div class="stat-card">
        <div class="stat-value" id="stat-vocab">–</div>
        <div class="stat-label">Từ khóa trong vocabulary</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-titles">–</div>
        <div class="stat-label">Tiêu đề phim / show</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-genres">–</div>
        <div class="stat-label">Loại từ khóa: genre</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-kw">–</div>
        <div class="stat-label">Từ khóa từ description</div>
      </div>
    </div>
    <div class="instructions">
      <h3>📐 Kiến trúc mô hình</h3>
      <ol>
        <li><strong>Keyword Suggestor</strong>: <code>TF-IDF char_wb ngram(2,4)</code> + <code>KNN cosine brute-force</code></li>
        <li><strong>Content Recommender</strong>: <code>TF-IDF word ngram(1,2), top-15k features</code> + <code>KNN cosine brute-force</code></li>
        <li><strong>Dataset</strong>: <code>netflix_titles.csv</code> – 3 cột sử dụng: <code>title</code>, <code>listed_in</code>, <code>description</code></li>
        <li><strong>Similarity score</strong>: <code>1 – cosine_distance</code> (0 = hoàn toàn khác, 1 = giống hệt)</li>
      </ol>
    </div>
  </div>
</main>

<script>
  const input = document.getElementById('searchInput');
  const dropdown = document.getElementById('dropdown');
  let debounceTimer = null;
  let currentQuery = '';

  // ── Debounced input handler ───────────────────────────────────────
  input.addEventListener('input', () => {
    const q = input.value.trim();
    clearTimeout(debounceTimer);
    if (q.length < 2) { dropdown.style.display = 'none'; return; }
    debounceTimer = setTimeout(() => fetchSuggestions(q), 180);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      dropdown.style.display = 'none';
      fetchRecommendations(input.value.trim());
    }
    if (e.key === 'Escape') dropdown.style.display = 'none';
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-wrap')) dropdown.style.display = 'none';
  });

  // ── Fetch keyword suggestions ─────────────────────────────────────
  async function fetchSuggestions(q) {
    currentQuery = q;
    try {
      const res = await fetch(`/api/suggest?q=${encodeURIComponent(q)}&k=8`);
      const data = await res.json();
      renderDropdown(data.suggestions || []);
      renderSuggestCards(data.suggestions || []);
    } catch(e) {
      dropdown.innerHTML = '<div class="dd-empty">Lỗi kết nối server</div>';
      dropdown.style.display = 'block';
    }
  }

  function renderDropdown(suggestions) {
    if (!suggestions.length) {
      dropdown.innerHTML = '<div class="dd-empty">Không tìm thấy gợi ý phù hợp</div>';
      dropdown.style.display = 'block';
      return;
    }
    dropdown.innerHTML = suggestions.map(s => `
      <div class="dd-item" onclick="selectSuggestion('${escHtml(s.keyword)}')">
        <span class="badge badge-${s.type === 'title' ? 'title' : s.type === 'genre' ? 'genre' : 'keyword'}">
          ${s.type}
        </span>
        <span class="dd-item-text">${escHtml(s.keyword)}</span>
        <span class="dd-item-score">${(s.score * 100).toFixed(1)}%</span>
      </div>`).join('');
    dropdown.style.display = 'block';
  }

  function renderSuggestCards(suggestions) {
    const container = document.getElementById('suggest-results');
    if (!suggestions.length) { container.innerHTML = ''; return; }
    container.innerHTML = suggestions.map(s => `
      <div class="card" onclick="selectSuggestion('${escHtml(s.keyword)}')">
        <div class="card-type">${s.type}</div>
        <div class="card-title">${escHtml(s.keyword)}</div>
        <span class="card-sim">Điểm tương đồng: ${(s.score * 100).toFixed(1)}%</span>
      </div>`).join('');
  }

  function selectSuggestion(keyword) {
    input.value = keyword;
    dropdown.style.display = 'none';
    fetchRecommendations(keyword);
    switchTab('recommend');
  }

  // ── Fetch content recommendations ────────────────────────────────
  async function fetchRecommendations(title) {
    if (!title) return;
    const container = document.getElementById('recommend-results');
    container.innerHTML = '<div style="color:var(--muted);padding:20px"><span class="spinner"></span>Đang tìm kiếm...</div>';
    switchTab('recommend');
    try {
      const res = await fetch(`/api/recommend?title=${encodeURIComponent(title)}&k=8`);
      const data = await res.json();
      renderRecommendCards(title, data.recommendations || []);
    } catch(e) {
      container.innerHTML = '<div style="color:var(--muted);padding:20px">Lỗi kết nối server.</div>';
    }
  }

  function renderRecommendCards(query, recs) {
    const container = document.getElementById('recommend-results');
    if (!recs.length) {
      container.innerHTML = `<div style="color:var(--muted);padding:20px">Không tìm thấy nội dung tương tự với "<strong>${escHtml(query)}</strong>".</div>`;
      return;
    }
    container.innerHTML = recs.map(r => `
      <div class="card" onclick="selectSuggestion('${escHtml(r.title)}')">
        <div class="card-type">${escHtml(r.type || '')} ${r.release_year ? '· ' + escHtml(r.release_year) : ''}</div>
        <div class="card-title">${escHtml(r.title)}</div>
        <div class="card-meta">${escHtml(r.listed_in)}</div>
        <div class="card-desc">${escHtml(r.description)}</div>
        <span class="card-sim">Độ tương đồng: ${(r.similarity * 100).toFixed(1)}%</span>
      </div>`).join('');
  }

  // ── Tab switcher ──────────────────────────────────────────────────
  function switchTab(name) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    document.querySelectorAll('.tab-btn')[['suggest','recommend','about'].indexOf(name)].classList.add('active');
    if (name === 'about') fetchStats();
  }

  // ── Stats ─────────────────────────────────────────────────────────
  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      const d = await res.json();
      document.getElementById('stat-vocab').textContent  = d.vocab_size?.toLocaleString() || '–';
      document.getElementById('stat-titles').textContent = d.title_count?.toLocaleString() || '–';
      document.getElementById('stat-genres').textContent = d.genre_count?.toLocaleString() || '–';
      document.getElementById('stat-kw').textContent     = d.keyword_count?.toLocaleString() || '–';
    } catch(e) {}
  }

  // ── Utilities ─────────────────────────────────────────────────────
  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }
</script>
</body>
</html>
"""


# ── API Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/suggest")
def api_suggest():
    """Gợi ý từ khóa khi gõ (autocomplete)."""
    q = request.args.get("q", "").strip()
    k = min(int(request.args.get("k", 8)), 20)
    if len(q) < 2:
        return jsonify({"suggestions": []})

    suggestions = suggestor.suggest(q, top_k=k)
    return jsonify({"query": q, "suggestions": suggestions})


@app.route("/api/recommend")
def api_recommend():
    """Gợi ý nội dung tương tự với một tiêu đề phim."""
    title = request.args.get("title", "").strip()
    k = min(int(request.args.get("k", 8)), 20)
    if not title:
        return jsonify({"recommendations": []})

    recommendations = recommender.recommend(title, top_k=k)
    return jsonify({"query": title, "recommendations": recommendations})


@app.route("/api/stats")
def api_stats():
    """Thống kê về mô hình đã huấn luyện."""
    sources = suggestor.vocab_sources
    return jsonify({
        "vocab_size":    len(suggestor.vocabulary),
        "title_count":   sum(1 for v in sources.values() if v == "title"),
        "genre_count":   sum(1 for v in sources.values() if v == "genre"),
        "keyword_count": sum(1 for v in sources.values() if v == "keyword"),
    })


# ── Chạy server ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 Flask server khởi động tại: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
