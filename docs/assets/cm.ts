import { continueRender, delayRender } from "remotion";

// Computer Modern (CMU) web fonts
const BASE = "https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/font";
export const CM_SERIF = "'Computer Modern Serif', serif";
export const CM_MONO = "'Computer Modern Typewriter', monospace";

if (typeof document !== "undefined" && !document.getElementById("cm-fonts")) {
  const handle = delayRender("Computer Modern");
  for (const [id, path] of [["cm-fonts", "Serif/cmun-serif.css"], ["cm-mono", "Typewriter/cmun-typewriter.css"]]) {
    const l = document.createElement("link");
    l.id = id; l.rel = "stylesheet"; l.href = `${BASE}/${path}`;
    document.head.appendChild(l);
  }
  setTimeout(() => {
    Promise.all(["32px 'Computer Modern Serif'", "italic 32px 'Computer Modern Serif'", "32px 'Computer Modern Typewriter'"]
      .map((f) => document.fonts.load(f))).finally(() => continueRender(handle));
  }, 300);
}
