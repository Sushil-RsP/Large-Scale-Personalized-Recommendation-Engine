// Frontend logic for Recommendation Engine Dashboard

let selectedGenres = new Set();
let currentVariant = null;

const GENRES = [
  "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  renderGenreChips();
  fetchSystemInfo();
  fetchABMetrics();
  fetchOfflineBenchmark();
});

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      if (targetId === "tab-ab") fetchABMetrics();
      if (targetId === "tab-benchmark") fetchOfflineBenchmark();
    });
  });
}

function renderGenreChips() {
  const container = document.getElementById("genreChips");
  container.innerHTML = "";
  GENRES.forEach(g => {
    const chip = document.createElement("div");
    chip.className = "genre-chip";
    chip.textContent = g;
    chip.onclick = () => {
      if (selectedGenres.has(g)) {
        selectedGenres.delete(g);
        chip.classList.remove("selected");
      } else {
        selectedGenres.add(g);
        chip.classList.add("selected");
      }
    };
    container.appendChild(chip);
  });
}

function toggleUserType() {
  const userType = document.querySelector('input[name="userType"]:checked').value;
  const warmGroup = document.getElementById("warmUserGroup");
  const coldGroup = document.getElementById("coldUserGroup");

  if (userType === "warm") {
    warmGroup.classList.remove("hidden");
    coldGroup.classList.add("hidden");
  } else {
    warmGroup.classList.add("hidden");
    coldGroup.classList.remove("hidden");
  }
}

async function fetchSystemInfo() {
  try {
    const res = await fetch("/system/info");
    const data = await res.json();
    document.getElementById("pillFaiss").textContent = `FAISS: ${data.faiss_total_indexed.toLocaleString()} Items`;
    document.getElementById("pillCache").textContent = `Cache: ${data.cache.backend === 'in_memory_lru' ? 'In-Memory LRU' : 'Redis'} (${(data.cache.hit_ratio * 100).toFixed(1)}%)`;
  } catch (err) {
    console.warn("Could not fetch system info:", err);
  }
}

async function runRecommendation() {
  const btn = document.getElementById("btnRecommend");
  btn.disabled = true;
  btn.textContent = "Inferencing...";

  const userType = document.querySelector('input[name="userType"]:checked').value;
  const customId = document.getElementById("customUserId").value.trim();
  const selectId = document.getElementById("userSelect").value;
  const targetUid = userType === "warm" ? (customId ? parseInt(customId) : parseInt(selectId)) : null;

  const rankingModel = document.getElementById("rankingModelSelect").value;
  const topK = parseInt(document.getElementById("topKInput").value) || 10;
  const bypassCache = document.getElementById("bypassCacheCheck").checked;

  const payload = {
    user_id: targetUid,
    top_k: topK,
    candidate_k: 100,
    ranking_model: rankingModel,
    onboarding_genres: Array.from(selectedGenres),
    bypass_cache: bypassCache
  };

  try {
    const resp = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    renderRecommendationResults(data);
    fetchSystemInfo();
  } catch (err) {
    alert("Recommendation request failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Recommendations";
  }
}

function renderRecommendationResults(data) {
  currentVariant = data.ab_test_variant || null;

  // Update telemetry stats
  document.getElementById("valTotalLatency").textContent = `${data.total_latency_ms.toFixed(1)} ms`;
  document.getElementById("valCacheStatus").textContent = data.cache_hit ? "CACHE HIT" : "CACHE MISS";
  document.getElementById("valStage1Candidates").textContent = data.candidates_retrieved_stage1 ? `${data.candidates_retrieved_stage1} Items` : "Fallback";
  document.getElementById("valModelUsed").textContent = data.ab_test_variant_name ? `${data.ab_test_variant_name}` : data.ranking_model_used;

  // Render Waterfall
  const wfContainer = document.getElementById("waterfallBars");
  wfContainer.innerHTML = "";
  const latencies = data.stage_latencies_ms || {};
  const total = Math.max(0.1, data.total_latency_ms);

  const stageLabels = {
    user_tower_inference_ms: "User Tower (ONNX)",
    faiss_retrieval_ms: "FAISS MIPS Search",
    stage1_retrieval_total_ms: "Stage 1 Candidate Retrieval",
    deep_ranking_inference_ms: "Deep Ranker Inference",
    stage2_ranking_total_ms: "Stage 2 Re-Ranking",
    cold_start_ms: "Cold-Start Fallback",
    total_pipeline_ms: "Total End-to-End"
  };

  for (const [key, val] of Object.entries(latencies)) {
    const row = document.createElement("div");
    row.className = "waterfall-row";

    const label = stageLabels[key] || key;
    const pct = Math.min(100, Math.max(4, (val / total) * 100));

    row.innerHTML = `
      <div class="wf-label">${label}</div>
      <div class="wf-bar-wrapper">
        <div class="wf-bar" style="width: ${pct}%;"></div>
      </div>
      <div class="wf-val">${val.toFixed(2)} ms</div>
    `;
    wfContainer.appendChild(row);
  }

  // Render Movie Cards
  const movieGrid = document.getElementById("movieGrid");
  movieGrid.innerHTML = "";
  const recs = data.recommendations || [];

  if (recs.length === 0) {
    movieGrid.innerHTML = `<div class="empty-state">No recommendations returned.</div>`;
    return;
  }

  document.getElementById("resultsMeta").textContent = `Showing Top ${recs.length} ${data.is_cold_start ? '(Cold-Start Fallback)' : '(Two-Stage Ranked)'}`;

  recs.forEach(movie => {
    const card = document.createElement("div");
    card.className = "movie-card";

    let deltaHtml = "";
    if (movie.rank_delta !== undefined) {
      if (movie.rank_delta > 0) {
        deltaHtml = `<span class="delta-pill delta-pos">▲ +${movie.rank_delta}</span>`;
      } else if (movie.rank_delta < 0) {
        deltaHtml = `<span class="delta-pill delta-neg">▼ ${movie.rank_delta}</span>`;
      } else {
        deltaHtml = `<span class="delta-pill delta-zero">= 0</span>`;
      }
    }

    const rankLabel = movie.final_rank ? `#${movie.final_rank}` : `#${recs.indexOf(movie) + 1}`;
    const scoreVal = movie.predicted_ctr !== undefined
      ? `P(Click): ${(movie.predicted_ctr * 100).toFixed(1)}%`
      : `Bayesian: ${movie.bayesian_score ? movie.bayesian_score.toFixed(2) : '--'}`;

    card.innerHTML = `
      <div>
        <div class="card-top">
          <span class="rank-pill">${rankLabel}</span>
          ${deltaHtml}
        </div>
        <h4 class="movie-title">${escapeHtml(movie.title)}</h4>
        <div class="movie-genres">${escapeHtml(movie.genres.replace(/\|/g, " • "))}</div>
      </div>
      <div>
        <div class="movie-stats">
          <span>★ ${movie.item_avg_rating ? movie.item_avg_rating.toFixed(2) : '--'}</span>
          <span class="ctr-score-badge">${scoreVal}</span>
        </div>
        <button class="btn-click-event" onclick="logClickEvent('${movie.movieId}', this)">
          Like / Click (Test CTR)
        </button>
      </div>
    `;
    movieGrid.appendChild(card);
  });
}

async function logClickEvent(movieId, btnElement) {
  btnElement.disabled = true;
  btnElement.textContent = "Clicked ✓";
  btnElement.style.borderColor = "var(--accent-green)";
  btnElement.style.color = "var(--accent-green)";

  if (currentVariant) {
    try {
      await fetch("/ab-test/event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ variant: currentVariant, event_type: "click" })
      });
      fetchABMetrics();
    } catch (e) {
      console.error(e);
    }
  }
}

async function fetchABMetrics() {
  try {
    const res = await fetch("/ab-test/metrics");
    const data = await res.json();

    const va = data.variant_A;
    const vb = data.variant_B;
    const analysis = data.analysis;

    document.getElementById("ctrA").textContent = `${(va.ctr * 100).toFixed(2)}%`;
    document.getElementById("impA").textContent = va.impressions.toLocaleString();
    document.getElementById("clkA").textContent = va.clicks.toLocaleString();
    document.getElementById("latA").textContent = `${va.median_latency_ms.toFixed(1)} ms`;

    document.getElementById("ctrB").textContent = `${(vb.ctr * 100).toFixed(2)}%`;
    document.getElementById("impB").textContent = vb.impressions.toLocaleString();
    document.getElementById("clkB").textContent = vb.clicks.toLocaleString();
    document.getElementById("latB").textContent = `${vb.median_latency_ms.toFixed(1)} ms`;

    document.getElementById("statLift").textContent = `${analysis.relative_ctr_uplift_pct > 0 ? '+' : ''}${analysis.relative_ctr_uplift_pct.toFixed(2)}%`;
    document.getElementById("statZ").textContent = analysis.z_statistic.toFixed(3);
    document.getElementById("statP").textContent = analysis.p_value.toFixed(4);
    document.getElementById("statCI").textContent = `[${analysis.ci_lower_diff.toFixed(3)}, ${analysis.ci_upper_diff.toFixed(3)}]`;
    document.getElementById("statConclusion").textContent = analysis.conclusion;
  } catch (err) {
    console.warn("Could not fetch A/B metrics:", err);
  }
}

async function simulateTraffic(count) {
  try {
    const res = await fetch(`/ab-test/simulate?count=${count}`, { method: "POST" });
    await res.json();
    fetchABMetrics();
  } catch (err) {
    alert("Simulation failed: " + err.message);
  }
}

async function fetchOfflineBenchmark() {
  try {
    const res = await fetch("/metrics/offline");
    const data = await res.json();
    if (data.Stage1_Retrieval_Only) {
      const tbody = document.getElementById("benchmarkTableBody");
      tbody.innerHTML = `
        <tr>
          <td><strong>Stage 1: Two-Tower FAISS (Retrieval Only)</strong></td>
          <td>${data.Stage1_Retrieval_Only['recall@50']}</td>
          <td>${data.Stage1_Retrieval_Only['ndcg@10']}</td>
          <td>${(data.Stage1_Retrieval_Only['hit_rate@10'] * 100).toFixed(2)}%</td>
          <td>${data.Stage1_Retrieval_Only['mrr']}</td>
          <td>${data.Stage1_Retrieval_Only['latency_ms']} ms</td>
        </tr>
        <tr>
          <td><strong>Stage 2: Two-Tower + Wide & Deep</strong></td>
          <td>${data.Stage2_Wide_and_Deep['recall@50']}</td>
          <td>${data.Stage2_Wide_and_Deep['ndcg@10']}</td>
          <td>${(data.Stage2_Wide_and_Deep['hit_rate@10'] * 100).toFixed(2)}%</td>
          <td>${data.Stage2_Wide_and_Deep['mrr']}</td>
          <td>${data.Stage2_Wide_and_Deep['latency_ms']} ms</td>
        </tr>
        <tr class="highlight-row">
          <td><strong>Stage 2: Two-Tower + DCN-v2 (Deep & Cross)</strong></td>
          <td>${data.Stage2_DCN_v2['recall@50']}</td>
          <td>${data.Stage2_DCN_v2['ndcg@10']}</td>
          <td>${(data.Stage2_DCN_v2['hit_rate@10'] * 100).toFixed(2)}%</td>
          <td>${data.Stage2_DCN_v2['mrr']}</td>
          <td>${data.Stage2_DCN_v2['latency_ms']} ms</td>
        </tr>
      `;
    }
  } catch (err) {
    console.warn("Could not fetch offline benchmark:", err);
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

