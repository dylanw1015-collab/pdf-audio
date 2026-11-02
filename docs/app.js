/* Read-Along: an offline player for EPUB 3 read-along books (Media Overlays).

Why this exists: Apple Books plays Media Overlays but has no speed control for
them. Here the narration is an HTML <audio> element, whose playbackRate changes
speed while keeping the voice's pitch (preservesPitch, on by default; see
https://html.spec.whatwg.org/multipage/media.html#dom-media-preservespitch).

How it works:
* Import: the .epub is unzipped once (JSZip) and every file is stored in
  IndexedDB, so the book opens later with no network, and only the pages on
  screen are held in memory.
* Sync: each SMIL <par> pairs a sentence id on a page with a clip of an audio
  file (EPUB Media Overlays 3.3, https://www.w3.org/TR/epub-33/#sec-media-overlays).
  While the audio plays, the par whose clip contains the current time gets the
  book's active class, which is what the book's CSS highlights.
* Pages render inside a shadow root each, so the book's stylesheet styles the
  page and nothing else.
*/
"use strict";

const $ = (s) => document.querySelector(s);
const audio = $("#audio");
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// ------------------------------------------------------------------ storage
let dbp;
function db() {
  dbp ||= new Promise((ok, bad) => {
    const r = indexedDB.open("readalong", 1);
    r.onupgradeneeded = () => {
      r.result.createObjectStore("books", { keyPath: "id" });
      r.result.createObjectStore("files");
    };
    r.onsuccess = () => ok(r.result);
    r.onerror = () => bad(r.error);
  });
  return dbp;
}
async function tx(storeName, mode, fn) {
  const d = await db();
  return new Promise((ok, bad) => {
    const t = d.transaction(storeName, mode);
    const req = fn(t.objectStore(storeName));
    t.oncomplete = () => ok(req && req.result);
    t.onerror = t.onabort = () => bad(t.error);
  });
}
const putFile = (key, blob) => tx("files", "readwrite", (s) => s.put(blob, key));
const getFile = (key) => tx("files", "readonly", (s) => s.get(key));

// ------------------------------------------------------------------ parsing
const XML = (s) => new DOMParser().parseFromString(s, "application/xml");
function resolve(base, href) {
  const u = new URL(href, "http://x/" + base);
  return decodeURIComponent(u.pathname.slice(1)) + u.hash;
}
function clock(v) {
  if (!v) return null;
  v = v.trim();
  const m = v.match(/^([\d.]+)(h|min|s|ms)?$/);
  if (m) {
    const n = parseFloat(m[1]);
    return { h: n * 3600, min: n * 60, ms: n / 1000 }[m[2]] ?? n;
  }
  return v.split(":").map(Number).reduce((a, b) => a * 60 + b, 0);
}
const byTag = (doc, tag) => [...doc.getElementsByTagNameNS("*", tag)];

async function importEpub(file, say) {
  say("Opening the book…");
  const zip = await JSZip.loadAsync(file);
  const read = (p) => {
    const f = zip.file(p);
    if (!f) throw new Error("Missing " + p + " in the book");
    return f.async("string");
  };
  const container = XML(await read("META-INF/container.xml"));
  const opfPath = byTag(container, "rootfile")[0].getAttribute("full-path");
  const opf = XML(await read(opfPath));
  const items = {};
  for (const it of byTag(opf, "item")) {
    items[it.getAttribute("id")] = {
      href: resolve(opfPath, it.getAttribute("href")),
      type: it.getAttribute("media-type") || "",
      mo: it.getAttribute("media-overlay"),
      props: it.getAttribute("properties") || "",
    };
  }
  const title = (byTag(opf, "title")[0]?.textContent || file.name.replace(/\.epub$/i, "")).trim();
  let activeClass = "-epub-media-overlay-active";
  for (const m of byTag(opf, "meta")) {
    if (m.getAttribute("property") === "media:active-class") activeClass = m.textContent.trim();
  }
  const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

  // Pages, in reading order.
  const pages = [], css = new Map(), pageOf = {};
  const spine = byTag(opf, "itemref").map((r) => items[r.getAttribute("idref")]).filter(Boolean);
  for (const [i, it] of spine.entries()) {
    if (i % 25 === 0) say(`Reading pages… ${i + 1} of ${spine.length}`);
    const doc = new DOMParser().parseFromString(await read(it.href), "application/xhtml+xml");
    const vp = [...doc.querySelectorAll("meta")].find((m) => m.getAttribute("name") === "viewport");
    const w = +(vp?.getAttribute("content").match(/width\s*=\s*(\d+)/) || [])[1] || 1000;
    const h = +(vp?.getAttribute("content").match(/height\s*=\s*(\d+)/) || [])[1] || Math.round(w * 1.3);
    for (const l of doc.querySelectorAll("link[rel~=stylesheet]")) {
      const p = resolve(it.href, l.getAttribute("href"));
      if (!css.has(p) && zip.file(p)) css.set(p, await read(p));
    }
    const body = doc.querySelector("body") || doc.documentElement;
    body.querySelectorAll("script").forEach((s) => s.remove());
    for (const el of body.querySelectorAll("[id]")) el.id = `p${i}-${el.id}`;
    for (const img of body.querySelectorAll("img[src], image")) {
      const attr = img.hasAttribute("src") ? "src" : "href";
      const raw = img.getAttribute(attr) || img.getAttributeNS("http://www.w3.org/1999/xlink", "href");
      if (raw) {
        img.setAttribute("data-src", resolve(it.href, raw));
        img.removeAttribute(attr);
      }
    }
    pageOf[it.href] = i;
    pages.push({ w, h, html: new XMLSerializer().serializeToString(body).replace(/^<body[^>]*>|<\/body>$/g, "") });
  }

  // Sentences and their audio clips, in reading order.
  say("Reading the narration timings…");
  const pars = [];
  for (const it of spine) {
    const mo = it.mo && items[it.mo];
    if (!mo) continue;
    const smil = XML(await read(mo.href));
    for (const par of byTag(smil, "par")) {
      const t = byTag(par, "text")[0], a = byTag(par, "audio")[0];
      if (!t || !a) continue;
      const [doc, frag] = resolve(mo.href, t.getAttribute("src")).split("#");
      if (!(doc in pageOf) || !frag) continue;
      pars.push({
        page: pageOf[doc], id: `p${pageOf[doc]}-${frag}`,
        audio: resolve(mo.href, a.getAttribute("src")),
        begin: clock(a.getAttribute("clipBegin")) ?? 0,
        end: clock(a.getAttribute("clipEnd")) ?? Infinity,
      });
    }
  }

  // Table of contents from the navigation document.
  const toc = [];
  const nav = Object.values(items).find((x) => x.props.split(" ").includes("nav"));
  if (nav) {
    const nd = new DOMParser().parseFromString(await read(nav.href), "application/xhtml+xml");
    const tocNav = [...nd.querySelectorAll("nav")].find((n) =>
      (n.getAttributeNS("http://www.idpf.org/2007/ops", "type") || n.getAttribute("epub:type") || "").includes("toc")) || nd.querySelector("nav");
    for (const a of tocNav ? tocNav.querySelectorAll("a[href]") : []) {
      const p = resolve(nav.href, a.getAttribute("href")).split("#")[0];
      if (p in pageOf) toc.push({ label: a.textContent.trim().replace(/\s+/g, " "), page: pageOf[p] });
    }
  }

  // Images and audio go into storage one file at a time (a whole book can be
  // hundreds of MB; holding it all at once can run a phone out of memory).
  const media = Object.values(items).filter((x) => /^(image|audio|video)\//.test(x.type) && zip.file(x.href));
  for (const [k, x] of media.entries()) {
    if (k % 10 === 0) say(`Saving pictures and audio… ${k + 1} of ${media.length}`);
    const data = await zip.file(x.href).async("arraybuffer");
    await putFile(`${id}/${x.href}`, new Blob([data], { type: x.type }));
  }
  const book = { id, title, added: Date.now(), size: file.size, pages, pars, toc, activeClass, css: [...css.values()].join("\n") };
  await tx("books", "readwrite", (s) => s.put(book));
  return book;
}

// ------------------------------------------------------------------ library
async function showLibrary() {
  stopReader();
  $("#reader").hidden = true;
  $("#library").hidden = false;
  document.title = "Read-Along";
  const books = (await tx("books", "readonly", (s) => s.getAll())) || [];
  const ul = $("#books");
  ul.replaceChildren();
  for (const b of books.sort((a, c) => (store.get("opened:" + c.id, c.added)) - (store.get("opened:" + a.id, a.added)))) {
    const li = document.createElement("li");
    const open = document.createElement("button");
    open.className = "open";
    const pos = store.get("pos:" + b.id, 0);
    const pct = b.pars.length ? Math.round((100 * pos) / b.pars.length) : 0;
    open.innerHTML = "<strong></strong><small></small>";
    open.querySelector("strong").textContent = b.title;
    open.querySelector("small").textContent = `${b.pages.length} pages · ${Math.round(b.size / 1e6)} MB · ${pct}% read`;
    open.onclick = () => openBook(b.id);
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "Remove";
    del.onclick = async () => {
      if (!confirm(`Remove "${b.title}" from this phone?`)) return;
      await tx("books", "readwrite", (s) => s.delete(b.id));
      await tx("files", "readwrite", (s) => s.delete(IDBKeyRange.bound(b.id + "/", b.id + "/￿")));
      showLibrary();
    };
    li.append(open, del);
    ul.append(li);
  }
}

$("#file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  const say = (m) => ($("#status").textContent = m);
  try {
    if (navigator.storage?.persist) navigator.storage.persist();
    const b = await importEpub(f, say);
    say(b.pars.length ? `Added "${b.title}".` : `Added "${b.title}", but it has no narration to play.`);
    showLibrary();
  } catch (err) {
    say("Couldn't add that file: " + err.message);
  }
});

// ------------------------------------------------------------------ reader
let book = null, cur = -1, curAudio = null, urls = new Map(), io = null, raf = 0;
let lastTouch = 0, wraps = [], loading = null;

function stopReader() {
  audio.pause();
  cancelAnimationFrame(raf);
  io?.disconnect();
  for (const u of urls.values()) URL.revokeObjectURL(u);
  urls.clear();
  book = null; cur = -1; curAudio = null;
  audio.removeAttribute("src");
}

async function blobUrl(path) {
  if (urls.has(path)) return urls.get(path);
  const blob = await getFile(`${book.id}/${path}`);
  const u = blob ? URL.createObjectURL(blob) : "";
  urls.set(path, u);
  return u;
}

async function openBook(bookId) {
  stopReader();
  book = await tx("books", "readonly", (s) => s.get(bookId));
  if (!book) return showLibrary();
  store.set("opened:" + book.id, Date.now());
  document.title = book.title;
  $("#library").hidden = true;
  $("#reader").hidden = false;

  const toc = $("#toc");
  toc.replaceChildren();
  const entries = book.toc.length ? book.toc : [{ label: book.title, page: 0 }];
  for (const [k, t] of entries.entries()) toc.add(new Option(t.label, k));

  const host = $("#pages");
  host.replaceChildren();
  wraps = book.pages.map((p, i) => {
    const w = document.createElement("div");
    w.className = "pagewrap";
    w.style.aspectRatio = `${p.w} / ${p.h}`;
    w.dataset.i = i;
    host.append(w);
    return w;
  });
  layout();
  io = new IntersectionObserver((es) => es.forEach((e) => (e.isIntersecting ? renderPage(+e.target.dataset.i) : unrenderPage(+e.target.dataset.i))),
    { rootMargin: "1200px 0px" });
  wraps.forEach((w) => io.observe(w));

  setRate(store.get("rate", 1));
  const pos = Math.min(store.get("pos:" + book.id, 0), book.pars.length - 1);
  if (book.pars.length) setCur(Math.max(pos, 0), true);
  mediaSession();
}

function layout() {
  if (!book) return;
  const W = $("#pages").clientWidth;
  for (const w of wraps) {
    const s = w.shadowRoot?.querySelector(".scaler");
    if (s) s.style.transform = `scale(${W / book.pages[+w.dataset.i].w})`;
  }
}
addEventListener("resize", layout);

async function renderPage(i) {
  const w = wraps[i];
  if (!book || w.dataset.on) return;
  w.dataset.on = "1";
  const p = book.pages[i];
  const root = w.shadowRoot || w.attachShadow({ mode: "open" });
  root.innerHTML = `<style>${book.css}\n:host{display:block}.scaler{position:absolute;left:0;top:0;transform-origin:0 0}</style>` +
    `<div class="scaler" style="width:${p.w}px;height:${p.h}px">${p.html}</div>`;
  root.querySelector(".scaler").style.transform = `scale(${$("#pages").clientWidth / p.w})`;
  for (const img of root.querySelectorAll("[data-src]")) {
    const u = await blobUrl(img.dataset.src);
    if (!w.dataset.on) return;
    img.tagName.toLowerCase() === "image" ? img.setAttribute("href", u) : (img.src = u);
  }
  if (cur >= 0 && book.pars[cur].page === i) mark(true);
}

function unrenderPage(i) {
  const w = wraps[i];
  if (!w || !w.dataset.on) return;
  delete w.dataset.on;
  for (const img of w.shadowRoot.querySelectorAll("[data-src]")) {
    const u = urls.get(img.dataset.src);
    if (u) { URL.revokeObjectURL(u); urls.delete(img.dataset.src); }
  }
  w.shadowRoot.innerHTML = "";
}

function elFor(k) {
  const p = book.pars[k];
  return wraps[p.page].shadowRoot?.getElementById(p.id) || null;
}
function mark(on) {
  const el = cur >= 0 && elFor(cur);
  if (el) el.classList.toggle(book.activeClass, on);
}

function setCur(k, scroll) {
  if (!book || k < 0 || k >= book.pars.length) return;
  mark(false);
  const turned = cur < 0 || book.pars[k].page !== book.pars[cur].page;
  cur = k;
  mark(true);
  store.set("pos:" + book.id, k);
  if (scroll || (turned && Date.now() - lastTouch > 4000)) follow();
  const t = book.toc.length ? book.toc.reduce((a, e, n) => (e.page <= book.pars[k].page ? n : a), 0) : 0;
  $("#toc").value = t;
}

// Keep the sentence being read in view: when it is off screen, scroll so it
// sits a third of the way down.
function follow() {
  const p = book.pars[cur];
  const el = elFor(cur);
  const r = (el?.firstElementChild || el)?.getBoundingClientRect();
  if (r && r.height && r.top > 70 && r.bottom < innerHeight - 150) return;
  const top = r && r.height ? scrollY + r.top - innerHeight / 3 : wraps[p.page].offsetTop - 60;
  scrollTo({ top: Math.max(0, top), behavior: Math.abs(top - scrollY) < innerHeight * 2 ? "smooth" : "auto" });
  if (!r || !r.height) setTimeout(() => cur >= 0 && elFor(cur) && follow(), 400);
}
for (const ev of ["touchstart", "wheel"]) addEventListener(ev, () => (lastTouch = Date.now()), { passive: true });

// ------------------------------------------------------------------ playback
async function playFrom(k) {
  if (!book || k < 0 || k >= book.pars.length) return;
  const p = book.pars[k];
  setCur(k, true);
  if (p.audio !== curAudio) {
    curAudio = p.audio;
    loading = new Promise((ok) => audio.addEventListener("loadedmetadata", ok, { once: true }));
    audio.src = await blobUrl(p.audio);
    await loading;
  }
  audio.currentTime = p.begin;
  applyRate();
  try { await audio.play(); } catch { /* needs a tap first */ }
}

function tick() {
  if (!book || cur < 0 || audio.paused || audio.seeking) return;
  const t = audio.currentTime, p = book.pars[cur], n = book.pars[cur + 1];
  if (n && n.audio === curAudio && n.begin >= p.begin && t >= n.begin - 0.01) return setCur(cur + 1);
  if (t >= p.end - 0.01) {
    if (!n) return audio.pause();
    // Clips that follow on in the same file just keep playing; anything else is a jump.
    if (n.audio !== curAudio || n.begin < p.end - 0.05 || n.begin - p.end > 1.5) playFrom(cur + 1);
  }
}
function loop() { tick(); raf = requestAnimationFrame(loop); }
audio.addEventListener("timeupdate", tick); // keeps going with the screen locked, when animation frames stop
audio.addEventListener("ended", () => { if (book && cur + 1 < book.pars.length) playFrom(cur + 1); });
audio.addEventListener("play", () => { applyRate(); cancelAnimationFrame(raf); loop(); playIcon(true); });
audio.addEventListener("pause", () => { cancelAnimationFrame(raf); playIcon(false); });
audio.addEventListener("loadedmetadata", applyRate);

function playIcon(on) {
  $("#playicon").setAttribute("d", on ? "M7 5h4v14H7zM13 5h4v14h-4z" : "M8 5v14l11-7z");
  $("#play").setAttribute("aria-label", on ? "Pause" : "Play");
  if ("mediaSession" in navigator) navigator.mediaSession.playbackState = on ? "playing" : "paused";
}

function toggle() {
  if (!book || !book.pars.length) return;
  if (!audio.paused) return audio.pause();
  const p = book.pars[cur];
  // Resume where it stopped if still inside this sentence, else start the sentence.
  if (p.audio === curAudio && audio.currentTime >= p.begin && audio.currentTime < p.end) return audio.play();
  playFrom(cur);
}
function step(d) {
  const k = Math.min(Math.max(cur + d, 0), book.pars.length - 1);
  audio.paused ? setCur(k, true) : playFrom(k);
}

// Speed. preservesPitch is the default, set anyway for older WebKit.
function applyRate() {
  const r = store.get("rate", 1);
  audio.defaultPlaybackRate = r;
  audio.playbackRate = r;
  audio.preservesPitch = audio.webkitPreservesPitch = true;
}
function setRate(r) {
  r = Math.round(Math.min(Math.max(+r, 0.5), 3) * 100) / 100;
  store.set("rate", r);
  $("#rate").value = r;
  $("#ratev").textContent = r.toFixed(2) + "×";
  applyRate();
}
$("#rate").addEventListener("input", (e) => setRate(e.target.value));

function mediaSession() {
  if (!("mediaSession" in navigator)) return;
  navigator.mediaSession.metadata = new MediaMetadata({ title: book.title, artist: "Read-Along" });
  const h = {
    play: () => toggle(), pause: () => audio.pause(),
    previoustrack: () => step(-1), nexttrack: () => step(1),
  };
  for (const [k, f] of Object.entries(h)) { try { navigator.mediaSession.setActionHandler(k, f); } catch { /* unsupported */ } }
}

// Tap a sentence to read from there.
$("#pages").addEventListener("click", (e) => {
  if (!book) return;
  const ids = new Set(e.composedPath().map((n) => n.id).filter(Boolean));
  const k = book.pars.findIndex((p) => ids.has(p.id));
  if (k < 0) return;
  audio.paused ? setCur(k) : playFrom(k);
});

$("#play").onclick = toggle;
$("#prev").onclick = () => step(-1);
$("#next").onclick = () => step(1);
$("#back").onclick = showLibrary;
$("#toc").addEventListener("change", (e) => {
  const page = (book.toc[+e.target.value] || { page: 0 }).page;
  const k = book.pars.findIndex((p) => p.page >= page);
  if (k >= 0) audio.paused ? setCur(k, true) : playFrom(k);
  else wraps[page]?.scrollIntoView();
});

if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
showLibrary();
