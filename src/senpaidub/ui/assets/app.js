// Senpai Dub UI helpers, shared by every page. Plain ES module, no build step.
//
//   import { api, ApiError, uploadFile, events, setTitle, fmtDuration, fmtBytes,
//            renderHeader, mascot, toast, folderPicker, el } from "/assets/app.js";
//
// Rules: server strings only ever go into the DOM as text (el() children, textContent).
// The session cookie is sent automatically by fetch/EventSource; the UI never sees the token.

const APP_NAME = "Senpai Dub";

/** Error thrown by api() and uploadFile(): `message` is the server's `error` text, `status` the HTTP status (0 = server unreachable). */
export class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

function errorMessage(status, data, fallbackText) {
  if (data && typeof data.error === "string" && data.error) return data.error;
  if (data && data.detail) {
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg) return String(data.detail[0].msg);
  }
  if (status === 401) return "This page isn't signed in. Open the link printed by `senpaidub serve`.";
  if (status === 0) return "Can't reach Senpai Dub. Is `senpaidub serve` still running?";
  if (fallbackText && fallbackText.length < 200) return fallbackText;
  return `Request failed (HTTP ${status}).`;
}

/** JSON request to the local server. `body` is sent as JSON (or as-is for FormData). Resolves to the parsed JSON (null for empty bodies); throws ApiError with the server's message on failure. */
export async function api(path, { method = "GET", body, signal } = {}) {
  const opts = { method, headers: { Accept: "application/json" }, credentials: "same-origin", signal };
  if (body !== undefined) {
    if (body instanceof FormData) {
      opts.body = body;
    } else {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new ApiError(errorMessage(0), 0);
  }
  const type = res.headers.get("content-type") || "";
  let data = null;
  let text = "";
  if (type.includes("application/json")) {
    data = await res.json().catch(() => null);
  } else {
    text = await res.text().catch(() => "");
  }
  if (!res.ok) throw new ApiError(errorMessage(res.status, data, text), res.status, data);
  return data;
}

/** Upload one File to POST /api/uploads with progress. onProgress(fraction 0–1, loaded, total). Resolves to the upload dict {id, name, size, kind}. Pass an AbortSignal as the third argument to cancel. */
export function uploadFile(file, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.withCredentials = true;
    xhr.responseType = "text";
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (e) => {
      if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total, e.loaded, e.total);
    };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) {
        if (onProgress) onProgress(1, file.size, file.size);
        resolve(data);
      } else {
        reject(new ApiError(errorMessage(xhr.status, data, xhr.responseText), xhr.status, data));
      }
    };
    xhr.onerror = () => reject(new ApiError(errorMessage(0), 0));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    if (signal) {
      if (signal.aborted) { xhr.abort(); return; }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}

/** Live updates over SSE (/api/events). onJob(job) and onBatch(batch) get the job/batch dicts; opts.onStatus("open"|"error") fires on (re)connect so a page can refetch state. Reconnects automatically. Returns a function that closes the stream. */
export function events(onJob, onBatch, { onStatus } = {}) {
  let source = null;
  let closed = false;
  let timer = null;
  let delay = 1000;

  const dispatch = (raw, fallbackType) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }
    const type = (msg && msg.type) || fallbackType;
    // The server may send either {"type":"job","job":{...}} or the bare dict.
    if (type === "job" && onJob) onJob(msg.job || msg);
    else if (type === "batch" && onBatch) onBatch(msg.batch || msg);
  };

  const connect = () => {
    if (closed) return;
    source = new EventSource("/api/events", { withCredentials: true });
    source.addEventListener("job", (e) => safe(() => dispatch(e.data, "job")));
    source.addEventListener("batch", (e) => safe(() => dispatch(e.data, "batch")));
    source.onmessage = (e) => safe(() => dispatch(e.data, null));
    source.onopen = () => { delay = 1000; if (onStatus) safe(() => onStatus("open")); };
    source.onerror = () => {
      if (onStatus) safe(() => onStatus("error"));
      // The browser retries by itself unless the stream is CLOSED (e.g. HTTP error); then we retry.
      if (source.readyState === EventSource.CLOSED) {
        source.close();
        if (!closed) {
          timer = setTimeout(connect, delay);
          delay = Math.min(delay * 2, 15000);
        }
      }
    };
  };
  connect();
  return () => {
    closed = true;
    clearTimeout(timer);
    if (source) source.close();
  };
}

function safe(fn) {
  try { fn(); } catch (err) { console.error(err); }
}

const FAVICONS = {
  default:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" fill="#141217"/><circle cx="16" cy="15" r="11" fill="#fff" stroke="#FFD23F" stroke-width="2"/><ellipse cx="12" cy="14" rx="2.2" ry="3" fill="#141217"/><ellipse cx="20" cy="14" rx="2.2" ry="3" fill="#141217"/><path d="M13 20 q3 2.5 6 0" stroke="#C8102E" stroke-width="2" fill="none" stroke-linecap="round"/></svg>',
  done:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" fill="#0F7A5A"/><path d="M7 17 l6 6 l12 -13" stroke="#fff" stroke-width="5" fill="none" stroke-linecap="square"/></svg>',
  failed:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" fill="#C8102E"/><path d="M9 9 L23 23 M23 9 L9 23" stroke="#fff" stroke-width="5" stroke-linecap="square"/></svg>',
};

function setFavicon(kind) {
  let link = document.querySelector('link[rel="icon"]');
  if (!link) {
    link = document.createElement("link");
    link.rel = "icon";
    document.head.appendChild(link);
  }
  link.type = "image/svg+xml";
  link.href = "data:image/svg+xml," + encodeURIComponent(FAVICONS[kind] || FAVICONS.default);
}

/** Set the tab title to "<text> — Senpai Dub" (just "Senpai Dub" when empty). Text starting with ✓ or ✗ also swaps the favicon to the done/failed variant. */
export function setTitle(text) {
  const t = (text || "").trim();
  document.title = t ? `${t} — ${APP_NAME}` : APP_NAME;
  setFavicon(t.startsWith("✓") ? "done" : t.startsWith("✗") ? "failed" : "default");
}

/** Format seconds as a clock ("7:42", "1:02:03"), or with opts.words as "12 min 31 s" / "1 h 2 min". Returns "—" for null/NaN. */
export function fmtDuration(s, { words = false } = {}) {
  if (s === null || s === undefined || !Number.isFinite(Number(s))) return "—";
  const total = Math.max(0, Math.round(Number(s)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  if (words) {
    if (h) return m ? `${h} h ${m} min` : `${h} h`;
    if (m) return sec ? `${m} min ${sec} s` : `${m} min`;
    return `${sec} s`;
  }
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** Format a byte count as "512 B", "34.2 MB", "1.3 GB". Returns "—" for null/NaN. */
export function fmtBytes(n) {
  if (n === null || n === undefined || !Number.isFinite(Number(n))) return "—";
  let v = Math.max(0, Number(n));
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return i === 0 ? `${v} B` : `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

/** Tiny DOM builder. attrs: class, text, html is NOT supported (by design), style (string or object), dataset (object), on<Event> (function), boolean true/false toggles an attribute, null/undefined skipped; other keys become attributes (aria-*, for, type…). Children: nodes, strings/numbers (as text), arrays, null/false skipped. */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class" || key === "className") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "value" && "value" in node) node.value = value;
    else if (key === "checked" || key === "disabled" || key === "selected") node[key] = Boolean(value);
    else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, String(value));
  }
  appendChildren(node, children);
  return node;
}

function appendChildren(node, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) appendChildren(node, child);
    else if (child instanceof Node) node.appendChild(child);
    else node.appendChild(document.createTextNode(String(child)));
  }
}

const REPO_URL = "https://github.com/keraliss/senpaidub";

const NAV = [
  { href: "/", label: "New dub" },
  { href: "/settings", label: "Settings" },
  { href: REPO_URL, label: "GitHub", external: true },
];

/** Render the poster header (SENPAI DUB 先輩吹替, nav links, badge) into `target`. opts.badge is the badge text; opts.badgeKind is "yellow" (default), "live" (accent) or "ink". */
export function renderHeader(target, { badge = "● ON AIR", badgeKind = "yellow" } = {}) {
  const here = window.location.pathname.replace(/\/+$/, "") || "/";
  const header = el(
    "header",
    { class: "site-header" },
    el(
      "a",
      { class: "brand", href: "/", "aria-label": "Senpai Dub, home" },
      el("span", { class: "brand-name", text: "SENPAI DUB" }),
      el("span", { class: "brand-ja", lang: "ja", "aria-hidden": "true", text: "先輩吹替" }),
    ),
    el(
      "nav",
      { class: "site-nav", "aria-label": "Main" },
      NAV.map((item) =>
        el("a", {
          href: item.href,
          "aria-current": !item.external && item.href === here ? "page" : null,
          target: item.external ? "_blank" : null,
          rel: item.external ? "noopener noreferrer" : null,
          text: item.label,
        }),
      ),
    ),
    badge ? el("span", { class: `badge${badgeKind && badgeKind !== "yellow" ? ` badge--${badgeKind}` : ""}`, text: badge }) : null,
  );
  target.replaceChildren(header);
  return header;
}

const MASCOT_BASE =
  '<rect x="58" y="110" width="16" height="30" fill="#141217"/>' +
  '<rect x="36" y="136" width="60" height="10" rx="5" fill="#141217"/>' +
  '<rect x="16" y="14" width="100" height="100" rx="50" fill="#FFFFFF" stroke="#141217" stroke-width="5"/>' +
  '<path d="M30 40 Q66 24 102 40" stroke="#141217" stroke-width="3" fill="none" stroke-dasharray="4 6"/>';

const MASCOT_FACES = {
  normal: [
    "Senpai, the mascot",
    '<ellipse cx="50" cy="66" rx="9" ry="13" fill="#141217"/><ellipse cx="82" cy="66" rx="9" ry="13" fill="#141217"/>' +
      '<circle cx="53" cy="60" r="4" fill="#FFFFFF"/><circle cx="85" cy="60" r="4" fill="#FFFFFF"/>' +
      '<path d="M56 88 Q66 98 76 88" stroke="#141217" stroke-width="4" fill="none" stroke-linecap="round"/>' +
      '<ellipse cx="36" cy="84" rx="8" ry="4" fill="#F7A8B8"/><ellipse cx="96" cy="84" rx="8" ry="4" fill="#F7A8B8"/>' +
      '<path d="M112 20 l6 -10 M120 32 l11 -4 M104 10 l1 -10" stroke="#141217" stroke-width="4" stroke-linecap="round"/>',
  ],
  concentrating: [
    "Senpai concentrating",
    '<path d="M40 64 L60 70 M92 64 L72 70" stroke="#141217" stroke-width="6" stroke-linecap="round"/>' +
      '<path d="M54 92 L78 92" stroke="#141217" stroke-width="5" stroke-linecap="round"/>' +
      '<path d="M104 46 q8 12 0 18 q-8 -6 0 -18z" fill="#6EC3F2" stroke="#141217" stroke-width="2"/>',
  ],
  celebrating: [
    "Senpai celebrating",
    '<path d="M40 68 Q50 56 60 68 M72 68 Q82 56 92 68" stroke="#141217" stroke-width="5" fill="none" stroke-linecap="round"/>' +
      '<path d="M50 84 Q66 104 82 84 Z" fill="#141217"/>' +
      '<ellipse cx="36" cy="84" rx="8" ry="4" fill="#F7A8B8"/><ellipse cx="96" cy="84" rx="8" ry="4" fill="#F7A8B8"/>' +
      '<path d="M8 20 l6 6 M4 40 l9 0 M120 10 l-5 7 M124 34 l-8 2" stroke="#FFD23F" stroke-width="5" stroke-linecap="round"/>',
  ],
};

/** The Senpai mascot as an inline SVG string: mood "normal" | "concentrating" | "celebrating". Static markup, safe for innerHTML (e.g. into a .mascot span). The same art is in /assets/senpai-<mood>.svg. */
export function mascot(mood = "normal") {
  const [label, face] = MASCOT_FACES[mood] || MASCOT_FACES.normal;
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="132" height="150" viewBox="0 0 132 150" role="img" aria-label="${label}">` +
    MASCOT_BASE + face + "</svg>"
  );
}

let toastStack = null;

/** Show a short message in the corner. kind: "info" (default) | "ok" | "warn" | "error". Errors stay 8 s, others 4 s; all can be closed. Returns the toast element. */
export function toast(msg, kind = "info") {
  if (!toastStack || !document.body.contains(toastStack)) {
    toastStack = el("div", { class: "toast-stack", "aria-live": "polite" });
    document.body.appendChild(toastStack);
  }
  const close = () => node.remove();
  const node = el(
    "div",
    { class: `toast toast--${kind}`, role: kind === "error" ? "alert" : "status" },
    el("span", { class: "toast__msg", text: msg }),
    el("button", { type: "button", class: "toast__close", "aria-label": "Dismiss", onClick: close, text: "×" }),
  );
  toastStack.appendChild(node);
  setTimeout(close, kind === "error" ? 8000 : 4000);
  return node;
}

const ICON_DIR =
  '<svg viewBox="0 0 24 24" fill="#FFD23F" stroke="#141217" stroke-width="2" aria-hidden="true"><path d="M2 6h7l2 2h11v12H2z"/></svg>';
const ICON_VIDEO =
  '<svg viewBox="0 0 24 24" fill="none" stroke="#4A4450" stroke-width="2" aria-hidden="true"><rect x="3" y="5" width="18" height="14"/><path d="M10 9l5 3-5 3z" fill="#4A4450"/></svg>';

function icon(svg) {
  const span = el("span", { "aria-hidden": "true", style: { display: "inline-flex" } });
  span.innerHTML = svg; // static markup defined above, never server data
  return span;
}

/** Modal folder browser over GET /api/fs. Resolves to the chosen folder path, or null if cancelled. opts.start: folder to open first; opts.title: heading. */
export function folderPicker({ start = null, title = "PICK A FOLDER" } = {}) {
  return new Promise((resolve) => {
    let current = null;
    let chosen = null;
    let reqId = 0;

    const titleId = `fp-title-${Math.random().toString(36).slice(2, 8)}`;
    const pathInput = el("input", {
      class: "input input--mono",
      type: "text",
      "aria-label": "Folder path",
      spellcheck: "false",
      autocomplete: "off",
    });
    const upBtn = el("button", { type: "button", class: "btn btn--small btn--secondary", text: "↑ Up", disabled: true });
    const rootsRow = el("div", { class: "row", role: "group", "aria-label": "Places" });
    const list = el("ul", { class: "fs-list", "aria-label": "Folder contents" });
    const errorBox = el("p", { class: "field__error", role: "alert", hidden: true });
    const summary = el("span", { class: "small muted spacer" });
    const chooseBtn = el("button", { type: "button", class: "btn btn--primary btn--small", text: "Use this folder", disabled: true });
    const cancelBtn = el("button", { type: "button", class: "btn btn--secondary btn--small", text: "Cancel" });

    const dialog = el(
      "dialog",
      { class: "modal", "aria-labelledby": titleId },
      el(
        "div",
        { class: "modal__head" },
        el("h2", { class: "modal__title", id: titleId, text: title }),
        el("button", { type: "button", class: "btn btn--small btn--on-ink", "aria-label": "Close", text: "✕", onClick: () => dialog.close() }),
      ),
      el(
        "div",
        { class: "modal__body" },
        rootsRow,
        el(
          "form",
          {
            class: "input-row",
            onSubmit: (e) => {
              e.preventDefault();
              const p = pathInput.value.trim();
              if (p) load(p);
            },
          },
          pathInput,
          el("button", { type: "submit", class: "btn btn--secondary", text: "Go" }),
        ),
        el("div", { class: "row" }, upBtn, summary),
        errorBox,
        list,
      ),
      el("div", { class: "modal__foot" }, cancelBtn, chooseBtn),
    );

    upBtn.addEventListener("click", () => { if (upBtn.dataset.parent) load(upBtn.dataset.parent); });
    cancelBtn.addEventListener("click", () => dialog.close());
    chooseBtn.addEventListener("click", () => { chosen = current; dialog.close(); });
    dialog.addEventListener("close", () => {
      dialog.remove();
      resolve(chosen);
    });

    async function load(path, focusList = false) {
      const id = ++reqId;
      list.setAttribute("aria-busy", "true");
      errorBox.hidden = true;
      try {
        const q = path ? `?path=${encodeURIComponent(path)}` : "";
        const data = await api(`/api/fs${q}`);
        if (id !== reqId) return;
        render(data);
        if (focusList) {
          const first = list.querySelector("button");
          (first || pathInput).focus();
        }
      } catch (err) {
        if (id !== reqId) return;
        errorBox.textContent = err.message;
        errorBox.hidden = false;
        if (current === null && path) load(null); // bad start folder: fall back to the default
      } finally {
        if (id === reqId) list.removeAttribute("aria-busy");
      }
    }

    function render(data) {
      current = data.path;
      pathInput.value = data.path || "";
      chooseBtn.disabled = !data.path;
      upBtn.disabled = !data.parent;
      upBtn.dataset.parent = data.parent || "";

      rootsRow.replaceChildren(
        ...(data.roots || []).map((r) =>
          el("button", { type: "button", class: "btn btn--small btn--secondary", title: r.path, onClick: () => load(r.path), text: r.name || r.path }),
        ),
      );

      const entries = data.entries || [];
      const dirs = entries.filter((e) => e.kind === "dir").length;
      const videos = entries.length - dirs;
      summary.textContent = `${dirs} folder${dirs === 1 ? "" : "s"} · ${videos} video${videos === 1 ? "" : "s"} here`;

      if (!entries.length) {
        list.replaceChildren(el("li", { class: "fs-empty", text: "Nothing here: no folders or videos." }));
        return;
      }
      list.replaceChildren(
        ...entries.map((entry) =>
          el(
            "li",
            {},
            entry.kind === "dir"
              ? el(
                  "button",
                  { type: "button", class: "fs-item fs-item--dir", onClick: () => load(entry.path, true) },
                  icon(ICON_DIR),
                  el("span", { class: "fs-item__name", text: entry.name }),
                  el("span", { class: "fs-item__size", "aria-hidden": "true", text: "›" }),
                )
              : el(
                  "div",
                  { class: "fs-item fs-item--video" },
                  icon(ICON_VIDEO),
                  el("span", { class: "fs-item__name", text: entry.name }),
                  el("span", { class: "fs-item__size", text: fmtBytes(entry.size) }),
                ),
          ),
        ),
      );
    }

    document.body.appendChild(dialog);
    dialog.showModal();
    load(start);
  });
}
