// Helpers shared by the Drop (index.html) and Setup (setup.html) pages. Plain ES module.
// Page-specific on purpose: the shared helpers live in app.js.

export const VIDEO_EXTS = [".mkv", ".mp4"];
export const SUB_EXTS = [".srt", ".ass", ".ssa"];

export const ENGINES = {
  qwen: { label: "QWEN", blurb: "Local AI that copies each original voice. Runs on your GPU." },
  edge: { label: "EDGE", blurb: "Free online Microsoft voices. Fast; needs internet." },
  google: { label: "GOOGLE", blurb: "Google Chirp 3 HD voices. Fast; needs an API key." },
};

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
/** Natural string compare: "ep2" sorts before "ep10". */
export function naturalCompare(a, b) {
  return collator.compare(String(a), String(b));
}

export function extOf(name) {
  const m = /\.[^./\\]+$/.exec(String(name || ""));
  return m ? m[0].toLowerCase() : "";
}
export const isVideoName = (name) => VIDEO_EXTS.includes(extOf(name));
export const isSubName = (name) => SUB_EXTS.includes(extOf(name));

/** Last path segment, for / or \ separators. */
export function basename(path) {
  return String(path || "").split(/[\\/]/).filter(Boolean).pop() || String(path || "");
}

export function stripExt(name) {
  return String(name || "").replace(/\.[^./\\]+$/, "");
}

/** Run fn(item, index) over items with at most `limit` in flight; resolves when all are done. fn must not throw. */
export async function mapLimit(items, limit, fn) {
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
}

/** "about 25 min" style text for minutes-per-episode, or "—". */
export function fmtMinutes(m) {
  if (m === null || m === undefined || !Number.isFinite(Number(m))) return "—";
  const v = Number(m);
  if (v < 1) return "under 1 min";
  if (v >= 90) return `~${(v / 60).toFixed(1)} h`;
  return `~${Math.round(v)} min`;
}

const PATHS_KEY = "senpaidub.paths.";

/** Query string (without "?") carrying a list of disk paths to /setup. Long lists go through sessionStorage. */
export function pathsQuery(paths) {
  const json = JSON.stringify(paths);
  const direct = `paths=${encodeURIComponent(json)}`;
  if (direct.length <= 6000) return direct;
  const key = Math.random().toString(36).slice(2, 10);
  try {
    sessionStorage.setItem(PATHS_KEY + key, json);
    return `pathskey=${key}`;
  } catch {
    return direct;
  }
}

/** Read the list of disk paths from /setup's query (paths= or pathskey=). Returns [] when absent or invalid. */
export function readPaths(params) {
  let raw = params.get("paths");
  if (!raw && params.get("pathskey")) {
    try { raw = sessionStorage.getItem(PATHS_KEY + params.get("pathskey")); } catch { raw = null; }
  }
  if (!raw) return [];
  try {
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list.filter((p) => typeof p === "string" && p) : [];
  } catch {
    return [];
  }
}

/** Remove characters that can't be in a folder name, like the server does for series names (approximation for the preview only). */
export function sanitizeName(name) {
  return String(name || "").replace(/[\\/:*?"<>|\u0000-\u001f]/g, "").replace(/\s+/g, " ").trim().replace(/^\.+|\.+$/g, "");
}
