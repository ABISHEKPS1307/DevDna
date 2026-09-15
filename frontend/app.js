/* DevDNA frontend – upload, DNA report, ask (local + AI RAG), file explorer. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  repoName: "",
  mode: "local",
  aiAvailable: false,
  tree: null,
  openDirs: new Set(),
  filter: "",
  matches: [],
  matchRels: new Set(),
};

/* ---------------- helpers ---------------- */
function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function setStatus(elem, msg, isError) {
  elem.textContent = msg;
  elem.classList.remove("hidden", "ok", "err");
  elem.classList.add(isError ? "err" : "ok");
}
function clearStatus(elem) { elem.classList.add("hidden"); }
/* Canonical repository-relative path: strip "RepoName/" prefix. */
function stripRepo(p) {
  return state.repoName && p.startsWith(state.repoName + "/")
    ? p.slice(state.repoName.length + 1) : p;
}

/* ---------------- health & mode toggle ---------------- */
async function fetchHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    state.aiAvailable = Boolean(data.ai_available);
  } catch { state.aiAvailable = false; }
  if (!state.aiAvailable) {
    $("aiModeBtn").classList.add("unavailable");
    $("aiModeBtn").title = "Set OPENAI_API_KEY (and install openai) to enable AI RAG";
  }
}
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.mode = btn.dataset.mode;
    document.querySelectorAll(".mode-btn").forEach((b) =>
      b.classList.toggle("active", b === btn));
  });
});

/* ---------------- upload (XHR for progress) ---------------- */
const dropzone = $("dropzone");
$("browseBtn").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (e) => uploadFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));

function uploadFile(file) {
  if (!file) return;
  if (!/\.zip$/i.test(file.name)) {
    setStatus($("uploadStatus"), "Please choose a .zip file.", true); return;
  }
  clearStatus($("uploadStatus"));
  $("progressWrap").classList.remove("hidden");
  $("progressFill").style.width = "0%";
  $("progressLabel").textContent = "Uploading… 0%";

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/upload-repository");
  xhr.upload.addEventListener("progress", (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    $("progressFill").style.width = pct + "%";
    $("progressLabel").textContent = pct < 100 ? `Uploading… ${pct}%` : "Analyzing repository…";
  });
  xhr.addEventListener("load", () => {
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch {}
    $("progressWrap").classList.add("hidden");
    if (xhr.status >= 200 && xhr.status < 300) {
      setStatus($("uploadStatus"), `✅ ${data.repository || "Repository"} scanned successfully.`);
      renderReport(data);
    } else {
      setStatus($("uploadStatus"), data.detail || "Upload failed. Please try again.", true);
    }
  });
  xhr.addEventListener("error", () => {
    $("progressWrap").classList.add("hidden");
    setStatus($("uploadStatus"), "Network error during upload.", true);
  });
  const fd = new FormData();
  fd.append("file", file, file.name);
  xhr.send(fd);
}

/* ---------------- DNA report ---------------- */
function renderReport(data) {
  state.repoName = data.repository || "";
  state.matches = []; state.matchRels = new Set();
  $("previewCard").classList.add("hidden");

  const stats = [
    ["Code Files", data.total_code_files ?? 0],
    ["Framework", data.framework || "Unknown"],
    ["Endpoints", data.endpoints ?? 0],
    ["AI Chunks", data.chunks_created ?? 0],
  ].map(([label, val]) => {
    const stat = el("div", "stat");
    stat.append(el("span", "stat-value", String(val)), el("span", "stat-label", label));
    return stat;
  });
  const statsWrap = el("div", "stats");
  stats.forEach((s) => statsWrap.append(s));

  const langs = el("div", "chips");
  const langEntries = Object.entries(data.languages || {});
  if (langEntries.length) {
    langEntries.forEach(([lang, n]) => langs.append(el("span", "chip", `${lang}: ${Number(n)}`)));
  } else {
    langs.append(el("span", "chip muted", "None detected"));
  }

  const fileList = el("ul", "file-list");
  const files = Array.isArray(data.detected_files) ? data.detected_files : [];
  if (files.length) files.forEach((f) => fileList.append(el("li", "", f)));
  else fileList.append(el("li", "muted", "None"));

  const report = $("reportCard");
  report.textContent = "";
  report.append(
    el("h2", "", "🚀 Repository DNA Report"),
    el("p", "repo-name", state.repoName || "Repository"),
    el("p", "scan-ok", "✅ Scanned successfully"),
    statsWrap,
    el("h3", "", "Languages"), langs,
    el("h3", "", "Detected Files"), fileList,
    el("h3", "", "Sample AI Chunk"),
  );
  const pre = el("pre", "code");
  pre.append(el("code", "", String(data.sample_chunk ?? "")));
  report.append(pre);

  $("workspace").classList.remove("hidden");
  $("explorerToggle").classList.remove("hidden");
  $("explorerRepo").textContent = state.repoName;
  report.scrollIntoView({ behavior: "smooth", block: "start" });
  loadFiles();
}

/* ---------------- file explorer ---------------- */
$("explorerToggle").addEventListener("click", () => {
  const open = $("explorer").classList.toggle("mobile-open");
  $("explorerToggle").setAttribute("aria-expanded", String(open));
});
$("treeFilter").addEventListener("input", (e) => {
  state.filter = e.target.value.trim().toLowerCase();
  renderTree();
});

async function loadFiles() {
  try {
    const res = await fetch("/repository/files");
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return;
    state.tree = Array.isArray(data.tree) ? data.tree : [];
    state.openDirs = new Set(
      state.tree.filter((n) => n.type === "dir").map((n) => n.path));
    renderTree();
  } catch { /* explorer is non-critical; ignore */ }
}

function filterTree(nodes, q) {
  const out = [];
  for (const node of nodes) {
    if (node.type === "file") {
      if (node.path.toLowerCase().includes(q)) out.push(node);
    } else {
      const kids = filterTree(node.children || [], q);
      if (kids.length || node.path.toLowerCase().includes(q)) {
        out.push({ ...node, children: kids });
      }
    }
  }
  return out;
}

function renderTree() {
  const container = $("fileTree");
  container.textContent = "";
  let tree = state.tree;
  if (!tree) { container.append(el("p", "muted small", "No files indexed.")); return; }
  if (state.filter) tree = filterTree(tree, state.filter);
  if (!tree.length) { container.append(el("p", "muted small", "No matching files.")); return; }
  container.append(buildNodes(tree, 0));
}

function buildNodes(nodes, depth) {
  const frag = document.createDocumentFragment();
  for (const node of nodes) {
    const row = el("div", `tree-row ${node.type}`);
    row.style.paddingLeft = 8 + depth * 14 + "px";
    if (node.type === "dir") {
      const chev = el("span", "chevron", "▸");
      row.append(chev, el("span", "dir-name", node.name));
      const kids = el("div", "tree-children");
      kids.append(buildNodes(node.children || [], depth + 1));
      const open = state.filter ? true : state.openDirs.has(node.path);
      kids.classList.toggle("open", open);
      if (open) chev.classList.add("rotated");
      row.addEventListener("click", () => {
        const isOpen = kids.classList.toggle("open");
        chev.classList.toggle("rotated", isOpen);
        if (isOpen) state.openDirs.add(node.path);
        else state.openDirs.delete(node.path);
      });
      frag.append(kids);
    } else {
      row.append(el("span", `file-badge ${extClass(node.name)}`, extLabel(node.name)),
                 el("span", "file-name", node.name));
      row.dataset.path = node.path;
      if (state.matchRels.has(node.path)) row.classList.add("hit");
      row.addEventListener("click", () => selectFile(node.path));
      frag.append(row);
    }
  }
  return frag;
}

function extOf(name) { const i = name.lastIndexOf("."); return i >= 0 ? name.slice(i + 1).toLowerCase() : ""; }
function extLabel(name) {
  const map = { py: "PY", js: "JS", ts: "TS", tsx: "TSX", jsx: "JSX",
    java: "JAVA", html: "HTML", htm: "HTML", css: "CSS", mjs: "JS" };
  return map[extOf(name)] || "•";
}
function extClass(name) {
  const ext = extOf(name);
  return ["py", "js", "ts", "tsx", "jsx", "java", "html", "css", "mjs"].includes(ext)
    ? `badge-${ext}` : "badge-other";
}

async function selectFile(path) {
  document.querySelectorAll(".tree-row.file.active").forEach((r) => r.classList.remove("active"));
  const row = document.querySelector(`.tree-row.file[data-path="${CSS.escape(path)}"]`);
  if (row) row.classList.add("active");

  const match = state.matches.find((m) => m.rel === path);
  if (match) {
    const details = document.querySelector(`details.code-item[data-rel="${CSS.escape(path)}"]`);
    if (details) {
      details.open = true;
      details.scrollIntoView({ behavior: "smooth", block: "center" });
      details.classList.remove("flash");
      void details.offsetWidth;
      details.classList.add("flash");
    }
    return;
  }
  try {
    const res = await fetch(`/repository/preview?path=${encodeURIComponent(path)}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { setStatus($("askStatus"), data.detail || "Could not preview file.", true); return; }
    clearStatus($("askStatus"));
    renderPreview(data);
  } catch {
    setStatus($("askStatus"), "Could not reach the server.", true);
  }
}

function renderPreview(data) {
  $("previewPath").textContent = data.path || "";
  $("previewLang").textContent = data.language || "Code";
  const body = $("previewBody");
  body.textContent = "";
  (Array.isArray(data.chunks) ? data.chunks : []).forEach((c) => {
    body.append(el("p", "chunk-divider", `Chunk ${Number(c.chunk ?? 0)}`));
    const pre = el("pre", "code");
    pre.append(el("code", "", String(c.content ?? "")));
    body.append(pre);
  });
  $("previewCard").classList.remove("hidden");
  $("previewCard").scrollIntoView({ behavior: "smooth", block: "start" });
}
$("previewClose").addEventListener("click", () => $("previewCard").classList.add("hidden"));

/* ---------------- ask ---------------- */
$("askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("questionInput").value.trim();
  if (!question) return;

  const btn = $("askBtn");
  btn.disabled = true; btn.textContent = "Thinking…";
  clearStatus($("askStatus"));

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: state.mode }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus($("askStatus"), data.detail || "Something went wrong.", true);
      return;
    }
    const matches = Array.isArray(data.matches) ? data.matches
      : Array.isArray(data.code_sections) ? data.code_sections : [];
    const answer = typeof data.answer === "string" && data.answer
      ? data.answer : "No answer was generated.";

    $("answerText").textContent = answer;   // textContent = XSS-safe
    $("modeBadge").textContent = data.mode_used === "ai" ? "AI RAG" : "Local Search";
    const notice = $("answerNotice");
    if (typeof data.notice === "string" && data.notice) {
      notice.textContent = data.notice;
      notice.classList.remove("hidden");
    } else {
      notice.classList.add("hidden");
    }
    $("answerBox").classList.remove("hidden");

    if (!matches.length) {
      $("codeSection").classList.add("hidden");
      state.matches = []; state.matchRels = new Set(); renderTree();
      setStatus($("askStatus"), "No supporting code found for this question.", true);
      return;
    }
    clearStatus($("askStatus"));
    state.matches = matches.map((m) => ({ ...m, rel: stripRepo(m.path) }));
    state.matchRels = new Set(state.matches.map((m) => m.rel));
    renderTree();
    renderMatches(state.matches, question);
  } catch {
    setStatus($("askStatus"), "Could not reach the server.", true);
  } finally {
    btn.disabled = false; btn.textContent = "Ask";
  }
});

function extractTerms(question) {
  const stop = new Set(["where", "what", "how", "which", "is", "are", "the", "a",
    "an", "in", "on", "of", "for", "and", "or", "does", "do", "did", "this",
    "that", "explain", "tell", "show", "find", "about", "code", "file", "me",
    "my", "please", "implemented", "initialize", "initialized", "used", "use"]);
  const raw = question.toLowerCase().match(/[a-z_][a-z0-9_]{1,}/g) || [];
  const terms = [...new Set(raw.filter((w) => !stop.has(w)))];
  terms.sort((a, b) => b.length - a.length);
  return terms.slice(0, 8);
}

function highlight(escapedCode, terms) {
  if (!terms.length) return escapedCode;
  const pattern = terms
    .map((t) => escapeHtml(t).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  try {
    return escapedCode.replace(new RegExp(pattern, "gi"), (m) => `<mark>${m}</mark>`);
  } catch { return escapedCode; }
}

function codeBlock(match, terms, isOpen) {
  const file = escapeHtml(match.file || "snippet");
  const meta = `${escapeHtml(match.path || match.file || "")} • Chunk ${Number(match.chunk ?? 0)} • Score ${Number(match.score ?? 0).toFixed(3)}`;
  const body = highlight(escapeHtml(match.content ?? ""), terms);
  return `
    <details class="code-item" ${isOpen ? "open" : ""} data-rel="${escapeHtml(match.rel || "")}">
      <summary><span class="file-chip">📄 ${file}</span><span class="meta">${meta}</span></summary>
      <pre class="code"><code>${body}</code></pre>
    </details>`;
}

function renderMatches(matches, question) {
  const terms = extractTerms(question);
  const [first, ...rest] = matches;
  $("primaryCode").innerHTML = codeBlock(first, terms, true);
  if (rest.length) {
    $("moreWrap").classList.remove("hidden");
    $("moreCode").innerHTML = rest.map((m) => codeBlock(m, terms, false)).join("");
  } else {
    $("moreWrap").classList.add("hidden");
    $("moreCode").innerHTML = "";
  }
  $("codeSection").classList.remove("hidden");
}

/* ---------------- init ---------------- */
fetchHealth();
