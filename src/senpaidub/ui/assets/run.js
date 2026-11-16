// Shared helpers for job.html and batch.html (live dubbing views). Plain ES module.
//
//   import { GROUPS, groupStates, stageMeta, renderStageCards, renderLog, coalesce,
//            liveElapsed, liveEta, openPath, statusChip, pct, jobTitleName } from "/assets/run.js";
//
// Server text only ever reaches the DOM through textContent / el().
import { api, el, fmtDuration, toast } from "/assets/app.js";

export const GROUPS = [
  { key: "subtitles", label: "Subtitles", ja: "字幕" },
  { key: "separate", label: "Split voices", ja: "分離" },
  { key: "voice", label: "Voices", ja: "声" },
  { key: "mix", label: "Mix", ja: "調整" },
  { key: "mux", label: "Pack", ja: "完成" },
];

/** 0-100 integer from a 0-1 progress (clamped). */
export function pct(p) {
  const n = Number(p);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.floor(n * 100)));
}

/** "done" | "active" | "todo" | "failed" for each group of a job. */
export function groupStates(job) {
  const idx = GROUPS.findIndex((g) => g.key === job.group);
  return GROUPS.map((g, i) => {
    if (job.status === "done" || job.status === "skipped") return "done";
    if (job.status === "waiting" || idx < 0) return "todo";
    if (i < idx) return "done";
    if (i === idx) {
      if (job.status === "failed") return "failed";
      if (job.status === "cancelled") return "todo";
      return "active";
    }
    return "todo";
  });
}

/** Short text for the current stage: "Voicing lines · 112 / 340" or just the label. */
export function stageMeta(job) {
  const label = job.stage_label || "";
  const done = job.stage_done;
  const total = job.stage_total;
  if (!label) return "";
  if (job.stage_unit === "seconds" && total) return `${label} · ${fmtDuration(done || 0)} / ${fmtDuration(total)}`;
  if (job.stage_unit === "items" && total) return `${label} · ${Math.floor(done || 0)} / ${Math.floor(total)}`;
  return label;
}

/** Build the 5 stage cards into `container` once; returns update(job). */
export function renderStageCards(container) {
  const cards = GROUPS.map((g, i) => {
    const num = el("span", { class: "stage-card__num", text: String(i + 1).padStart(2, "0") });
    const label = el("span", { class: "stage-card__label", text: g.label });
    const meta = el("span", { class: "stage-card__meta" });
    const node = el("li", { class: "stage-card stage-card--todo" }, num, label, meta);
    return { node, meta, state: "" };
  });
  container.replaceChildren(...cards.map((c) => c.node));
  return function update(job) {
    const states = groupStates(job);
    cards.forEach((c, i) => {
      const s = states[i];
      if (c.state !== s) {
        c.state = s;
        c.node.className = `stage-card stage-card--${s === "active" ? "active" : s}`;
        if (s === "active") c.node.setAttribute("aria-current", "step");
        else c.node.removeAttribute("aria-current");
      }
      const text = s === "done" ? "Done" : s === "active" ? stageMeta(job) : s === "failed" ? "Stopped here" : "Up next";
      if (c.meta.textContent !== text) c.meta.textContent = text;
    });
  };
}

/** Fill a <pre>/<div> log body with the job log (last 50), warnings highlighted; keeps scroll pinned to the bottom if it was. */
export function renderLog(body, log) {
  const entries = Array.isArray(log) ? log : [];
  const pinned = body.scrollTop + body.clientHeight >= body.scrollHeight - 24;
  if (!entries.length) {
    body.replaceChildren(el("span", { class: "log__empty", text: "Nothing yet." }));
    return;
  }
  body.replaceChildren(
    ...entries.map((e) =>
      el("span", { class: `log__line${e.level === "warn" ? " log__line--warn" : ""}`, text: `${e.level === "warn" ? "! " : ""}${e.msg}` }),
    ),
  );
  if (pinned) body.scrollTop = body.scrollHeight;
}

/** Coalesce calls: fn(latest) runs at most once per `ms`. Returns push(value); push.flush() runs now. */
export function coalesce(fn, ms = 250) {
  let timer = null;
  let latest;
  let has = false;
  const run = () => {
    timer = null;
    if (!has) return;
    has = false;
    fn(latest);
  };
  const push = (v) => {
    latest = v;
    has = true;
    if (timer === null) timer = setTimeout(run, ms);
  };
  push.flush = () => { clearTimeout(timer); run(); };
  return push;
}

/** Elapsed seconds, ticking locally between server updates. `stamp` is performance.now() when the job dict arrived. */
export function liveElapsed(job, stamp) {
  if (job.elapsed_s == null) return null;
  const extra = job.status === "running" ? (performance.now() - stamp) / 1000 : 0;
  return job.elapsed_s + extra;
}
/** ETA seconds counting down locally; null when unknown. */
export function liveEta(job, stamp) {
  if (job.eta_s == null || job.status !== "running") return null;
  return Math.max(0, job.eta_s - (performance.now() - stamp) / 1000);
}

/** POST /api/open for "play" or "reveal"; toasts on failure. */
export async function openPath(path, action) {
  if (!path) { toast("There is no file to open yet.", "warn"); return; }
  try {
    await api("/api/open", { method: "POST", body: { path, action } });
  } catch (err) {
    toast(err.message, "error");
  }
}

const CHIPS = {
  waiting: ["WAITING", "chip--outline"],
  running: ["DUBBING", "chip--accent"],
  done: ["DONE", "chip--ok"],
  failed: ["FAILED", "chip--accent"],
  skipped: ["ALREADY DUBBED", "chip--yellow"],
  cancelled: ["CANCELLED", "chip--outline"],
};
/** A status chip element. */
export function statusChip(status) {
  const [text, cls] = CHIPS[status] || [String(status || "").toUpperCase(), "chip--outline"];
  return el("span", { class: `chip ${cls}`, text });
}

/** "Ep 07" from a file name when it looks like one, else the name without extension. */
export function jobTitleName(name) {
  const base = String(name || "").replace(/\.[^.]+$/, "");
  const m = base.match(/(?:^|[\s._-])(?:e|ep|episode)\s*0*(\d{1,3})(?!\d)/i) || base.match(/(?:^|[\s._-])-\s*0*(\d{1,3})(?!\d)/);
  return m ? `Ep ${m[1].padStart(2, "0")}` : base;
}
