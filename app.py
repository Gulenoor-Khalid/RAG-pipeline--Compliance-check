"""
app.py — Construction Compliance Micro-Inspector
User uploads their own PDFs → live RAG → Groq answers
"""

import streamlit as st
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(
    page_title="Construction Compliance AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
h1,h2,h3 { color: #f0f6fc !important; }
h2 { font-size:1.05rem; font-weight:600; border-bottom:1px solid #30363d; padding-bottom:0.3rem; margin-bottom:0.7rem; }
.card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem 1.2rem; margin:0.4rem 0; }
.card-code    { border-left: 3px solid #388bfd; }
.card-project { border-left: 3px solid #f78166; }
.badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; }
.badge-pass { background:#1f4a2e; color:#3fb950; border:1px solid #3fb950; }
.badge-fail { background:#4a1f1f; color:#f85149; border:1px solid #f85149; }
.badge-warn { background:#4a3a1f; color:#d29922; border:1px solid #d29922; }
.badge-unknown { background:#2d333b; color:#8b949e; border:1px solid #8b949e; }
.chunk-meta { font-size:0.7rem; color:#8b949e; margin-bottom:0.4rem; }
.chunk-text { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#c9d1d9; line-height:1.6; white-space:pre-wrap; }
.answer-box  { background:#161b22; border:1px solid #3fb95055; border-radius:8px; padding:1.2rem 1.4rem; }
.answer-text { color:#e6edf3; line-height:1.8; font-size:0.93rem; }
.eval-panel  { background:#0d1117; border:1px solid #388bfd33; border-radius:6px; padding:0.8rem 1rem; margin:0.3rem 0; }
.eval-label  { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#388bfd; font-weight:600; margin-bottom:0.3rem; }
.metric-card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:0.9rem; text-align:center; }
.metric-value { font-size:1.7rem; font-weight:700; color:#f0f6fc; }
.metric-label { font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em; margin-top:2px; }
.upload-box { background:#161b22; border:2px dashed #30363d; border-radius:10px; padding:1.5rem; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def chunk_card(chunk, score, rank):
    cls  = "card-code" if chunk.document_type == "codebook" else "card-project"
    icon = "📘" if chunk.document_type == "codebook" else "📋"
    st.markdown(f"""
    <div class="card {cls}">
      <div class="chunk-meta">
        <b>#{rank}</b> &nbsp;{icon} <b>{chunk.document_type.upper()}</b>
        &nbsp;·&nbsp; id:<code>{chunk.chunk_id}</code>
        &nbsp;·&nbsp; {chunk.discipline}
        &nbsp;·&nbsp; {chunk.source_doc} p.{chunk.page}
        &nbsp;·&nbsp; score <b>{score:.4f}</b>
      </div>
      <div class="chunk-text">{chunk.text[:500]}</div>
    </div>""", unsafe_allow_html=True)


def compliance_card(cc):
    if cc is None: return
    cls = {"PASS":"badge-pass","FAIL":"badge-fail",
           "WARNING":"badge-warn","UNKNOWN":"badge-unknown"}.get(cc.status,"badge-unknown")
    sev_icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢","N/A":"⚪"}.get(cc.severity,"⚪")
    border_col = {"PASS":"#3fb950","FAIL":"#f85149","WARNING":"#d29922"}.get(cc.status,"#8b949e")
    st.markdown(f"""
    <div class="card" style="border-left:3px solid {border_col};">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <span class="badge {cls}">{cc.status}</span>
        <span style="font-size:0.82rem;color:#8b949e;">Severity: {sev_icon} {cc.severity}</span>
      </div>
      <div style="display:flex;gap:2rem;margin-bottom:8px;">
        <div><span style="color:#8b949e;font-size:0.72rem;">REQUIRED</span><br/><b style="color:#f0f6fc;">{cc.rule_value or '—'}</b></div>
        <div><span style="color:#8b949e;font-size:0.72rem;">ACTUAL</span><br/><b style="color:#f0f6fc;">{cc.project_value or '—'}</b></div>
      </div>
      <div style="color:#c9d1d9;font-size:0.85rem;">{cc.explanation}</div>
    </div>""", unsafe_allow_html=True)


def score_bar(label, value, color="#388bfd"):
    pct = min(value, 100)
    st.markdown(f"""
    <div style="margin:0.35rem 0;">
      <div style="display:flex;justify-content:space-between;">
        <span style="font-size:0.76rem;color:#8b949e;">{label}</span>
        <span style="font-size:0.76rem;color:#f0f6fc;font-weight:600;">{value:.1f}</span>
      </div>
      <div style="background:#21262d;border-radius:4px;height:7px;margin-top:3px;">
        <div style="width:{pct}%;height:7px;border-radius:4px;background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## 🏗️ Compliance Inspector")
        st.markdown("---")

        # Load API key from environment
        api_key = os.getenv("GROQ_API_KEY")

        st.markdown("---")
        st.markdown("### 📂 Upload Documents")

        code_file = st.file_uploader(
            "📘 Codebook PDF",
            type=["pdf","txt"],
            help="Building code, inspection reference, or regulatory document",
        )
        project_file = st.file_uploader(
            "📋 Specification PDF",
            type=["pdf","txt"],
            help="Project specification, scope of works, or technical drawings",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Retrieval Settings")
        top_k      = st.slider("Top-K chunks", 2, 10, 5)
        use_hybrid = st.toggle("Hybrid search (BM25 + Vector)", value=True)

        st.markdown("### 🗂️ Metadata Filter")
        doc_filter  = st.selectbox("Document type", ["All","codebook","specification"])
        disc_filter = st.selectbox("Discipline", ["All","fire_safety","structural","ventilation","electrical","plumbing","general"])

        st.markdown("---")
        eval_mode = st.toggle("🧪 Eval Mode", value=False)

        meta = {}
        if doc_filter  != "All": meta["document_type"] = doc_filter
        if disc_filter != "All": meta["discipline"]     = disc_filter

        return api_key, code_file, project_file, top_k, use_hybrid, meta, eval_mode


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
      <span style="font-size:2rem;">🏗️</span>
      <div>
        <h1 style="margin:0;font-size:1.5rem;font-weight:700;color:#f0f6fc;">
          Construction Compliance Inspector
        </h1>
        <p style="margin:0;color:#8b949e;font-size:0.85rem;">
          Upload your own documents · Hybrid RAG · Groq LLM · Compliance Engine
        </p>
      </div>
    </div>""", unsafe_allow_html=True)

    api_key, code_file, project_file, top_k, use_hybrid, meta, eval_mode = sidebar()

    # ── Upload gate ──────────────────────────────────────────────────────────
    if not code_file or not project_file:
        st.markdown("""
        <div class="upload-box">
          <div style="font-size:2.5rem;">📂</div>
          <div style="color:#f0f6fc;font-size:1rem;font-weight:600;margin:8px 0;">Upload your two documents to get started</div>
          <div style="color:#8b949e;font-size:0.85rem;">
            <b>Left panel →</b><br/>
            📘 <b>Codebook PDF</b> — regulatory document (building code, inspection reference, standards…)<br/>
            📋 <b>Specification PDF</b> — project document (scope of works, technical specifications, drawings…)
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Build index (cache by file content hash) ─────────────────────────────
    import hashlib as hl
    code_file.seek(0);    code_bytes    = code_file.read();    code_file.seek(0)
    project_file.seek(0); project_bytes = project_file.read(); project_file.seek(0)
    cache_key = hl.md5(code_bytes + project_bytes).hexdigest()

    if st.session_state.get("index_key") != cache_key:
        with st.status("📥 Processing your documents…", expanded=True) as status:
            from ingestion import ingest_uploaded, _get_embedder, EMBEDDING_MODEL
            from retrieval import HybridRetriever
            from reranker  import Reranker

            st.write("📄 Extracting text from PDFs…")
            # Pre-load model with visible feedback
            try:
                st.write(f"🤖 Loading embedding model ({EMBEDDING_MODEL}) — downloading ~80 MB on first run…")
                _get_embedder()
                st.write("✅ Embedding model ready")
            except ImportError as e:
                st.error(str(e))
                st.stop()

            st.write("✂️ Chunking and embedding documents…")
            chunks    = ingest_uploaded(code_file, project_file)
            st.write(f"🗂️ Building FAISS + BM25 index over {len(chunks)} chunks…")
            retriever = HybridRetriever()
            retriever.build(chunks)
            reranker  = Reranker()
            status.update(label="✅ Documents indexed and ready", state="complete", expanded=False)

            st.session_state["chunks"]    = chunks
            st.session_state["retriever"] = retriever
            st.session_state["reranker"]  = reranker
            st.session_state["index_key"] = cache_key

        st.success(f"✅ Indexed {len(chunks)} chunks  "
                   f"({sum(1 for c in chunks if c.document_type=='codebook')} codebook · "
                   f"{sum(1 for c in chunks if c.document_type=='specification')} specification)")

    chunks    = st.session_state["chunks"]
    retriever = st.session_state["retriever"]
    reranker  = st.session_state["reranker"]

    # ── Stats row ────────────────────────────────────────────────────────────
    codebook_n = sum(1 for c in chunks if c.document_type=="codebook")
    spec_n = sum(1 for c in chunks if c.document_type=="specification")
    disc_set = {c.discipline for c in chunks}
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-bottom:1rem;flex-wrap:wrap;">
      <div class="metric-card" style="flex:1;min-width:120px;">
        <div class="metric-value">{len(chunks)}</div>
        <div class="metric-label">Total Chunks</div>
      </div>
      <div class="metric-card" style="flex:1;min-width:120px;">
        <div class="metric-value" style="color:#388bfd;">{codebook_n}</div>
        <div class="metric-label">Codebook Chunks</div>
      </div>
      <div class="metric-card" style="flex:1;min-width:120px;">
        <div class="metric-value" style="color:#f78166;">{spec_n}</div>
        <div class="metric-label">Specification Chunks</div>
      </div>
      <div class="metric-card" style="flex:1;min-width:120px;">
        <div class="metric-value">{len(disc_set)}</div>
        <div class="metric-label">Disciplines</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Query ────────────────────────────────────────────────────────────────
    query = st.text_input("Ask a question about your documents",
                          placeholder="e.g. What is the minimum fire exit width?  |  Does the project comply with concrete cover rules?",
                          label_visibility="collapsed")

    col_btn, col_badge = st.columns([1,5])
    with col_btn:
        run = st.button("🔍 Analyse", use_container_width=True)
    with col_badge:
        if eval_mode:
            st.markdown('<span style="background:#1f4a2e;color:#3fb950;padding:4px 12px;border-radius:12px;font-size:0.78rem;font-weight:600;">🧪 EVAL MODE ON</span>', unsafe_allow_html=True)
        if not api_key:
            st.markdown('<span style="background:#4a1f1f;color:#f85149;padding:4px 12px;border-radius:12px;font-size:0.78rem;">⚠️ No Groq key — retrieval works but no AI answer</span>', unsafe_allow_html=True)

    if not run or not query.strip():
        return

    # ── Pipeline ─────────────────────────────────────────────────────────────
    t0 = time.time()

    raw_results      = retriever.retrieve(query, top_k=top_k, metadata_filter=meta or None, use_hybrid=use_hybrid)
    reranked, scores = reranker.rerank(query, raw_results)
    context_chunks   = [c for c, _ in reranked[:top_k]]

    from llm       import generate_answer, is_compliance_question
    from evaluator import evaluate

    with st.spinner("Generating answer…"):
        result = generate_answer(query, context_chunks, api_key=api_key or None)

    eval_result = evaluate(query, result.answer, context_chunks)
    elapsed     = time.time() - t0

    # ── Layout ───────────────────────────────────────────────────────────────
    if eval_mode:
        left, right = st.columns([3, 2])
    else:
        left  = st.container()
        right = None

    with left:
        # Compliance check
        if is_compliance_question(query) and result.compliance_check:
            st.markdown("#### ⚖️ Compliance Check")
            compliance_card(result.compliance_check)

        # Answer
        st.markdown("#### 💬 Answer")
        st.markdown(f'<div class="answer-box"><div class="answer-text">{result.answer.replace(chr(10),"<br>")}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#8b949e;font-size:0.74rem;">⏱ {elapsed:.2f}s · {len(context_chunks)} chunks · {"Hybrid" if use_hybrid else "Vector"} retrieval</span>', unsafe_allow_html=True)

        # Retrieved chunks
        st.markdown("---")
        st.markdown("#### 📄 Retrieved Chunks (post-rerank)")
        for i, (chunk, score) in enumerate(reranked[:top_k], 1):
            chunk_card(chunk, score, i)

    # ── Eval panel ────────────────────────────────────────────────────────────
    if eval_mode and right:
        with right:
            st.markdown("#### 🧪 Evaluation")

            risk_col = "#f85149" if eval_result.hallucination_risk > 50 else "#d29922" if eval_result.hallucination_risk > 25 else "#3fb950"
            faith_col = "#3fb950" if eval_result.faithfulness_score > 75 else "#d29922" if eval_result.faithfulness_score > 50 else "#f85149"
            
            # Main metrics grid
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
              <div class="metric-card">
                <div class="metric-value" style="color:#3fb950;">{eval_result.retrieval_score:.1f}</div>
                <div class="metric-label">Retrieval Score</div>
              </div>
              <div class="metric-card">
                <div class="metric-value" style="color:{risk_col};">{eval_result.hallucination_risk:.1f}</div>
                <div class="metric-label">Hallucination Risk</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Detailed metrics
            score_bar("Retrieval Quality",    eval_result.retrieval_score,    "#3fb950")
            score_bar("Faithfulness Score",  eval_result.faithfulness_score, faith_col)
            score_bar("Citation Coverage",   eval_result.citation_coverage,  "#388bfd")
            score_bar("Context Utilization", eval_result.context_utilization, "#d29922")
            score_bar("Retrieval Precision", eval_result.retrieval_precision, "#f78166")
            score_bar("Retrieval Hit Rate",  eval_result.retrieval_hit_rate,  "#7c3aed")
            score_bar("Hallucination Risk",  eval_result.hallucination_risk,  risk_col)

            # Correctness
            if eval_result.is_correct is not None:
                badge = "badge-pass" if eval_result.is_correct else "badge-fail"
                label = "CORRECT" if eval_result.is_correct else "INCORRECT"
                gold  = eval_result.matched_golden
                st.markdown(f"""
                <div class="eval-panel" style="margin-top:8px;">
                  <div class="eval-label">Correctness (Golden Dataset)</div>
                  <span class="badge {badge}">{label}</span>
                  <div style="color:#c9d1d9;font-size:0.8rem;margin-top:6px;">{eval_result.correctness_note}</div>
                  {"<div style='color:#8b949e;font-size:0.74rem;margin-top:4px;'>Expected: <i>" + gold.expected_answer + "</i></div>" if gold else ""}
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="eval-panel" style="margin-top:8px;">
              <div class="eval-label">Metadata Filter</div>
              <code style="color:#c9d1d9;font-size:0.78rem;">{meta if meta else "None"}</code>
            </div>""", unsafe_allow_html=True)

            with st.expander("📊 Before vs After Reranking"):
                st.markdown("**Before (RRF order):**")
                for i,(c,s) in enumerate(raw_results[:top_k],1):
                    st.markdown(f"`#{i}` [{c.document_type}] `{c.chunk_id}` rrf=`{s:.4f}`")
                    st.caption(c.text[:100]+"…")
                st.markdown("---")
                st.markdown("**After (multi-signal):**")
                sm = {s.chunk_id:s for s in scores}
                for i,(c,s) in enumerate(reranked[:top_k],1):
                    rs = sm.get(c.chunk_id)
                    detail = f"comp={rs.compliance_signal:.2f} num={rs.numeric_density:.2f} olap={rs.query_overlap:.2f}" if rs else ""
                    st.markdown(f"`#{i}` `{c.chunk_id}` final=`{s:.4f}` {detail}")
                    st.caption(c.text[:100]+"…")
            with st.expander("📤 Context Sent to LLM"):
                for i,c in enumerate(context_chunks,1):
                    st.markdown(f"**Chunk {i}** `{c.chunk_id}` {c.document_type} · {c.discipline} · p.{c.page}")
                    st.code(c.text[:300], language=None)

            with st.expander("📝 Full Prompt"):
                st.code(result.prompt, language="markdown")


if __name__ == "__main__":
    main()
