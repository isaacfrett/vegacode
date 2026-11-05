// The design suite's measurement script (src/vega/design_render.py passes it to the renderer as
// the body of an async function; `__OPTIONS__` is replaced with the JSON options first). It runs
// in an isolated script world: it shares the page's DOM, not the page's globals, so page script
// cannot change what it measures. It returns facts only; src/vega/design_render.py grades them.
const OPTIONS = __OPTIONS__;
const MAX_ITEMS = 600;

// -- colours: any CSS colour to sRGB bytes through a 1x1 canvas -------------------------------
const canvas = document.createElement('canvas');
canvas.width = canvas.height = 1;
const ctx = canvas.getContext('2d', {willReadFrequently: true});
const colourCache = new Map();
function rgba(css) {
  if (colourCache.has(css)) return colourCache.get(css);
  ctx.clearRect(0, 0, 1, 1);
  ctx.fillStyle = 'rgba(0, 0, 0, 0)';
  ctx.fillStyle = css;
  ctx.fillRect(0, 0, 1, 1);
  const d = ctx.getImageData(0, 0, 1, 1).data;
  const value = [d[0], d[1], d[2], Math.round(d[3] / 255 * 1000) / 1000];
  colourCache.set(css, value);
  return value;
}
function over(top, bottom) {  // `top` composited over opaque `bottom`
  const a = top[3];
  return [0, 1, 2].map(i => Math.round(top[i] * a + bottom[i] * (1 - a))).concat([1]);
}

// -- element helpers ---------------------------------------------------------------------------
const root = document.documentElement;
// Classic (non-overlay) scrollbars, shown when a mouse is attached, would take ~15px of the layout
// width on some machines and not others: hide them so every machine measures the same viewport.
const noScrollbars = document.createElement('style');
noScrollbars.textContent = '::-webkit-scrollbar { display: none; width: 0; height: 0; }';
(document.head || root).appendChild(noScrollbars);
void root.offsetWidth;
const vw = root.clientWidth || window.innerWidth, vh = window.innerHeight;
function box(r) {
  return {x: Math.round(r.left * 10) / 10, y: Math.round((r.top + window.scrollY) * 10) / 10,
          w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10};
}
function shown(el) {
  for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
    const s = getComputedStyle(e);
    if (s.display === 'none' || s.visibility === 'hidden' || s.visibility === 'collapse' ||
        parseFloat(s.opacity) === 0 || e.hidden) return false;
    if (s.contentVisibility === 'hidden') return false;
  }
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}
function srOnly(el) {  // the "visually hidden" pattern: clipped to <= 1px
  const r = el.getBoundingClientRect();
  return r.width <= 1.5 || r.height <= 1.5;
}
function describe(el) {
  if (!el || el.nodeType !== 1) return String(el && el.nodeName || '');
  let out = el.tagName.toLowerCase();
  if (el.id) return out + '#' + el.id;
  const classes = [...el.classList].slice(0, 2);
  if (classes.length) out += '.' + classes.join('.');
  const parent = el.parentElement;
  if (parent && parent !== document.body && parent !== root) {
    const same = [...parent.children].filter(c => c.tagName === el.tagName);
    if (same.length > 1) out += ':nth-of-type(' + (same.indexOf(el) + 1) + ')';
    return describe(parent) + ' > ' + out;
  }
  return out;
}
function text(el, limit = 80) { return (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, limit); }
function contentName(el) {  // name from content: text and img alts, skipping aria-hidden parts
  let out = '';
  for (const node of el.childNodes) {
    if (node.nodeType === 3) out += node.textContent;
    else if (node.nodeType === 1 && node.getAttribute('aria-hidden') !== 'true') {
      out += node.tagName === 'IMG' ? ' ' + (node.getAttribute('alt') || '') + ' ' : ' ' + contentName(node) + ' ';
    }
  }
  return out.replace(/\s+/g, ' ').trim();
}

// the page's own background (html, else body by propagation, else the Canvas colour of the root's
// colour scheme)
const probe = document.createElement('div');
probe.style.cssText = 'position:absolute;width:1px;height:1px;background:Canvas;color:CanvasText';
root.appendChild(probe);
const canvasColour = rgba(getComputedStyle(probe).backgroundColor);
probe.remove();
function pageBackground() {
  for (const e of [root, document.body]) {
    if (!e) continue;
    const c = rgba(getComputedStyle(e).backgroundColor);
    if (c[3] > 0) return over(c, canvasColour);
  }
  return canvasColour;
}
const pageBg = pageBackground();

// Effective background behind `el`: its ancestors' background colours composited from the root
// down. Gradients count by their worst stop; an image background is unknown.
const GRADIENT_COLOUR = /(rgba?\([^)]*\)|#[0-9a-f]{3,8}\b|color\([^)]*\)|oklch\([^)]*\)|hsla?\([^)]*\))/gi;
function backgrounds(el) {
  const chain = [];
  for (let e = el; e && e.nodeType === 1 && e !== root; e = e.parentElement) chain.push(e);
  chain.reverse();
  // pageBg already holds the root's background, or the body's when it propagated to the canvas
  const bodyPropagated = rgba(getComputedStyle(root).backgroundColor)[3] === 0;
  let layers = [pageBg];  // candidate backgrounds (several when a gradient is in the chain)
  let image = false;
  for (const e of chain) {
    const s = getComputedStyle(e);
    if (!(e === document.body && bodyPropagated)) {
      const c = rgba(s.backgroundColor);
      if (c[3] > 0) layers = layers.map(l => over(c, l));
    }
    const img = s.backgroundImage;
    if (img && img !== 'none') {
      if (/url\(/.test(img)) image = true;
      const stops = (img.match(GRADIENT_COLOUR) || []).map(rgba);
      if (stops.length) layers = stops.flatMap(stop => layers.map(l => over(stop, l)));
    }
  }
  return {layers: layers.slice(0, 12), image};
}
function opacity(el) {
  let o = 1;
  for (let e = el; e && e.nodeType === 1; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity);
  return o;
}

// -- text runs ------------------------------------------------------------------------------------
const walker = document.createTreeWalker(document.body || root, NodeFilter.SHOW_TEXT);
const runs = new Map();  // element -> {rect union of its own text}
for (let node = walker.nextNode(); node; node = walker.nextNode()) {
  if (!node.textContent.trim()) continue;
  const el = node.parentElement;
  if (!el || /^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE|TITLE|OPTION)$/.test(el.tagName)) continue;
  const range = document.createRange();
  range.selectNodeContents(node);
  const rects = [...range.getClientRects()].filter(r => r.width > 0 && r.height > 0);
  if (!rects.length) continue;
  const prev = runs.get(el) || {l: Infinity, t: Infinity, r: -Infinity, b: -Infinity};
  for (const r of rects) {
    prev.l = Math.min(prev.l, r.left); prev.t = Math.min(prev.t, r.top);
    prev.r = Math.max(prev.r, r.right); prev.b = Math.max(prev.b, r.bottom);
  }
  runs.set(el, prev);
}
const texts = [];
const textEls = [];
for (const [el, u] of runs) {
  if (texts.length >= MAX_ITEMS) break;
  if (!shown(el) || srOnly(el)) continue;
  const s = getComputedStyle(el);
  const fg = rgba(s.color);
  const bg = backgrounds(el);
  const o = opacity(el);
  texts.push({
    el: describe(el), text: text(el).slice(0, 40),
    rect: box(new DOMRect(u.l, u.t, u.r - u.l, u.b - u.t)),
    size: parseFloat(s.fontSize), weight: parseInt(s.fontWeight, 10) || 400,
    fg: [fg[0], fg[1], fg[2], Math.round(fg[3] * o * 1000) / 1000], bg: bg.layers, bg_image: bg.image,
    family: s.fontFamily.split(',')[0].trim(), disabled: !!el.closest(':disabled, [aria-disabled=true]'),
  });
  textEls.push(el);
}
// what is typed in a field is text too (it is not a text node)
for (const el of document.querySelectorAll('input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=range]):not([type=color]), textarea, select')) {
  if (texts.length >= MAX_ITEMS) break;
  const value = el.tagName === 'SELECT' ? (el.selectedOptions[0] ? el.selectedOptions[0].text : '') : el.value;
  if (!value || !value.trim() || !shown(el)) continue;
  const s = getComputedStyle(el);
  const fg = rgba(s.color);
  const o = opacity(el);
  texts.push({el: describe(el), text: value.slice(0, 40), rect: box(el.getBoundingClientRect()),
              size: parseFloat(s.fontSize), weight: parseInt(s.fontWeight, 10) || 400,
              fg: [fg[0], fg[1], fg[2], Math.round(fg[3] * o * 1000) / 1000], bg: backgrounds(el).layers,
              bg_image: backgrounds(el).image, family: s.fontFamily.split(',')[0].trim(), disabled: el.disabled, field: true});
}

// -- interactive elements and focus ---------------------------------------------------------------
const FOCUSABLE = 'a[href], button, input:not([type=hidden]), select, textarea, summary, ' +
                  '[tabindex], [contenteditable=""], [contenteditable=true], [role=button], [role=link]';
const controls = [...document.querySelectorAll(FOCUSABLE)].filter(e => shown(e) && !srOnly(e)).slice(0, 300);
function inlineLink(el) {  // a link inside a sentence: exempt from target size (WCAG 2.5.8)
  if (el.tagName !== 'A' || getComputedStyle(el).display !== 'inline') return false;
  const parent = el.parentElement;
  return !!parent && text(parent).length > text(el).length + 3;
}
function labelBox(el) {
  if (!/^(INPUT)$/.test(el.tagName) || !/^(checkbox|radio)$/.test(el.type)) return null;
  const label = el.closest('label') || (el.id && document.querySelector('label[for="' + CSS.escape(el.id) + '"]'));
  return label && shown(label) ? label.getBoundingClientRect() : null;
}
function accessibleName(el) {
  const by = el.getAttribute('aria-labelledby');
  if (by) return by.split(/\s+/).map(id => document.getElementById(id)).filter(Boolean).map(e => text(e)).join(' ').trim();
  if (el.getAttribute('aria-label')) return el.getAttribute('aria-label').trim();
  if (el.labels && el.labels.length) return [...el.labels].map(l => text(l)).join(' ').trim();
  if (el.tagName === 'IMG') return (el.getAttribute('alt') || '').trim();
  if (el.getAttribute('title')) return el.getAttribute('title').trim();
  const role = el.getAttribute('role') || '';
  const fromContent = /^(A|BUTTON|SUMMARY|H[1-6]|TH|TD|OPTION|LEGEND|CAPTION)$/.test(el.tagName) ||
                      /^(button|link|tab|menuitem|heading|cell|columnheader|rowheader|checkbox|radio|switch|option|treeitem)$/.test(role);
  if (fromContent) {
    const own = contentName(el);
    if (own) return own;
  }
  if (el.tagName === 'INPUT' && /^(submit|button|reset)$/.test(el.type)) return el.value || '';
  if (/^(INPUT|TEXTAREA)$/.test(el.tagName) && el.placeholder) return '';  // placeholder is not a label
  return '';
}

// Keyboard focus styling without needing a focused window: every author rule using :focus,
// :focus-visible or :focus-within is copied right after itself with those pseudo-classes replaced
// by an attribute (same specificity, same place in the cascade, same @media), and the browser's
// own focus ring is emulated with a zero-specificity rule ahead of every author sheet. Setting
// the attribute on one element then shows exactly the styles keyboard focus would give it.
const MARK = 'data-vega-kbfocus';
let rulesRead = 0, rulesUnreadable = 0;
function rewrite(sel) {
  return sel.replace(/:focus-visible/g, '[' + MARK + ']')
            .replace(/:focus-within/g, ':has([' + MARK + '])')
            .replace(/:focus(?![\w-])/g, '[' + MARK + ']')
            .replace(/:-moz-focusring/g, '[' + MARK + ']');
}
function copyFocusRules(list) {
  for (let i = 0; i < list.cssRules.length; i++) {
    const rule = list.cssRules[i];
    if (rule.cssRules && !(rule instanceof CSSStyleRule)) {  // @media, @supports, @layer blocks
      try { copyFocusRules(rule); } catch (e) { rulesUnreadable++; }
      continue;
    }
    if (!(rule instanceof CSSStyleRule) || !/:focus|:-moz-focusring/.test(rule.selectorText)) continue;
    const copy = rewrite(rule.selectorText) + '{' + rule.style.cssText + '}';
    try { list.insertRule(copy, i + 1); i++; rulesRead++; } catch (e) { rulesUnreadable++; }
  }
}
const ua = document.createElement('style');
ua.textContent = ':where([' + MARK + ']) { outline: auto 3px -webkit-focus-ring-color; }';
(document.head || root).insertBefore(ua, (document.head || root).firstChild);
for (const sheet of [...document.styleSheets]) {
  if (sheet.ownerNode === ua) continue;
  try { copyFocusRules(sheet); } catch (e) { rulesUnreadable++; }
}
const FOCUS_PROPS = ['outline-style', 'outline-width', 'outline-color', 'outline-offset', 'box-shadow',
                     'border-top-color', 'border-bottom-color', 'border-top-width', 'border-bottom-width',
                     'background-color', 'color', 'text-decoration-line', 'text-decoration-thickness'];
function focusStyle(el) {
  const s = getComputedStyle(el);
  const out = {};
  for (const p of FOCUS_PROPS) out[p] = s.getPropertyValue(p);
  return out;
}
const focusables = [];
for (const el of controls) {
  const tabindex = el.getAttribute('tabindex');
  const r = el.getBoundingClientRect();
  const label = labelBox(el);
  const target = label ? {w: Math.max(r.width, label.width), h: Math.max(r.height, label.height)} : {w: r.width, h: r.height};
  const before = focusStyle(el);
  el.setAttribute(MARK, '');
  const after = focusStyle(el);
  el.removeAttribute(MARK);
  const changed = FOCUS_PROPS.filter(p => before[p] !== after[p]);
  const outlineOn = after['outline-style'] !== 'none' && (after['outline-style'] === 'auto' ||
                    parseFloat(after['outline-width']) >= 1) && rgba(after['outline-color'])[3] > 0;
  focusables.push({
    el: describe(el), tag: el.tagName.toLowerCase(), text: text(el).slice(0, 40),
    rect: box(r), target: {w: Math.round(target.w * 10) / 10, h: Math.round(target.h * 10) / 10},
    inline_link: inlineLink(el), tabindex: tabindex === null ? null : parseInt(tabindex, 10),
    disabled: !!el.disabled, name: accessibleName(el),
    focus: {changed, outline: outlineOn, outline_style: after['outline-style'], outline_colour: rgba(after['outline-color']),
            shadow: after['box-shadow'] !== 'none' && after['box-shadow'] !== before['box-shadow'],
            bg: backgrounds(el.parentElement || el).layers[0]},
  });
}
ua.remove();
window.__vegaFocusables = controls;

// -- overlap: text runs and controls of unrelated elements drawn on top of each other -----------
const ignore = OPTIONS.overlap_ignore || '';
function ignored(el) { try { return !!(ignore && el.closest(ignore)); } catch (e) { return false; } }
function lineRects(el) {
  const out = [];
  for (const node of el.childNodes) {
    if (node.nodeType !== 3 || !node.textContent.trim()) continue;
    const range = document.createRange();
    range.selectNodeContents(node);
    for (const r of range.getClientRects()) if (r.width > 0 && r.height > 0 && out.length < 24) out.push(r);
  }
  return out;
}
const items = [];
for (const el of textEls) if (!ignored(el)) items.push({el, rects: lineRects(el)});
for (const el of controls) {
  if (inlineLink(el) || ignored(el)) continue;
  items.push({el, rects: [el.getBoundingClientRect()]});
}
for (const el of document.querySelectorAll('img, svg, video, canvas')) {
  if (shown(el) && !ignored(el) && !el.closest('svg *')) items.push({el, rects: [el.getBoundingClientRect()]});
}
const overlaps = [];
for (let i = 0; i < items.length && overlaps.length < 10; i++) {
  for (let j = i + 1; j < items.length && overlaps.length < 10; j++) {
    const a = items[i], b = items[j];
    if (a.el === b.el || a.el.contains(b.el) || b.el.contains(a.el)) continue;
    let hit = null;
    for (const ra of a.rects) {
      for (const rb of b.rects) {
        const w = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
        const h = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
        // tight line-heights make neighbouring lines' glyph boxes touch: a real overlap covers a
        // good share of the smaller box
        if (w > 2 && h > 2 && h > 0.35 * Math.min(ra.height, rb.height) && w > 0.2 * Math.min(ra.width, rb.width)) {
          hit = {w: Math.round(w), h: Math.round(h)}; break;
        }
      }
      if (hit) break;
    }
    if (hit) overlaps.push({a: describe(a.el), b: describe(b.el), w: hit.w, h: hit.h});
  }
}

// -- clipped: text cut off by an ancestor that hides overflow (scrollable ones do not count, and
// neither does deliberate single-line truncation with text-overflow: ellipsis)
const clipped = [];
for (const el of textEls) {
  if (clipped.length >= 6) break;
  if (ignored(el)) continue;
  const rects = lineRects(el);
  if (!rects.length) continue;
  const bottom = Math.max(...rects.map(r => r.bottom)), right = Math.max(...rects.map(r => r.right));
  for (let a = el; a && a !== document.body && a !== root; a = a.parentElement) {
    const s = getComputedStyle(a);
    if (s.textOverflow === 'ellipsis') break;
    const hidesY = /hidden|clip/.test(s.overflowY), hidesX = /hidden|clip/.test(s.overflowX);
    if (!hidesY && !hidesX) continue;
    const r = a.getBoundingClientRect();
    if (srOnly(a)) break;
    if ((hidesY && bottom > r.bottom + 2) || (hidesX && right > r.right + 2)) {
      clipped.push({el: describe(el), by: describe(a), text: text(el).slice(0, 30)});
      break;
    }
  }
}

// -- structure ------------------------------------------------------------------------------------
function styledVisible(el) {  // not hidden by CSS (an image that failed to load has no size yet still counts)
  for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
    const s = getComputedStyle(e);
    if (s.display === 'none' || s.visibility === 'hidden' || e.hidden) return false;
  }
  return true;
}
const images = [...document.querySelectorAll('img, [role=img], svg')].filter(e => e.tagName === 'IMG' ? styledVisible(e) : shown(e)).slice(0, 200).map(el => ({
  el: describe(el), tag: el.tagName.toLowerCase(), has_alt: el.tagName === 'IMG' ? el.hasAttribute('alt') : null,
  alt: el.tagName === 'IMG' ? (el.getAttribute('alt') || '') : (el.getAttribute('aria-label') || ''),
  hidden: el.getAttribute('aria-hidden') === 'true' || el.getAttribute('role') === 'presentation',
  rect: box(el.getBoundingClientRect()),
}));
const headings = [...document.querySelectorAll('h1, h2, h3, h4, h5, h6, [role=heading]')].filter(shown).map(el => ({
  el: describe(el), level: el.getAttribute('role') === 'heading' ? parseInt(el.getAttribute('aria-level') || '2', 10)
                                                                  : parseInt(el.tagName[1], 10),
  text: text(el).slice(0, 60),
}));
const formControls = [...document.querySelectorAll('input:not([type=hidden]):not([type=submit]):not([type=button]), select, textarea')]
  .filter(shown).map(el => ({el: describe(el), name: accessibleName(el), type: el.type || el.tagName.toLowerCase()}));

// -- spacing --------------------------------------------------------------------------------------
const spacing = [];
if (OPTIONS.spacing) {
  const PROPS = ['margin-top', 'margin-bottom', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left'];
  for (const el of [...document.querySelectorAll(OPTIONS.spacing)].filter(shown).slice(0, 300)) {
    const s = getComputedStyle(el);
    const values = {};
    for (const p of PROPS) values[p] = s.getPropertyValue(p);
    if (/flex|grid/.test(s.display)) { values['row-gap'] = s.rowGap; values['column-gap'] = s.columnGap; }
    spacing.push({el: describe(el), values});
  }
}

// -- probes for the case's functional checks ----------------------------------------------------
const probes = {};
for (const p of OPTIONS.probes || []) {
  let found;
  try { found = [...document.querySelectorAll(p.selector)]; } catch (e) { probes[p.selector] = {error: String(e)}; continue; }
  probes[p.selector] = {items: found.slice(0, 100).map(el => {
    const s = getComputedStyle(el);
    const styles = {};
    for (const prop of p.styles || []) styles[prop] = s.getPropertyValue(prop).trim();
    const attrs = {};
    for (const a of p.attrs || []) attrs[a] = el.getAttribute(a);
    return {el: describe(el), shown: shown(el), rect: box(el.getBoundingClientRect()), text: text(el, 4000),
            styles, attrs, name: accessibleName(el)};
  })};
}
const rootStyles = {};
for (const prop of OPTIONS.root_props || []) rootStyles[prop] = getComputedStyle(root).getPropertyValue(prop).trim();

// -- overflow ---------------------------------------------------------------------------------------
const scrollWidth = root.scrollWidth;
const wide = [];
for (const el of document.body ? document.body.querySelectorAll('*') : []) {
  if (wide.length >= 8) break;
  const r = el.getBoundingClientRect();
  if (r.right > vw + 1 && r.width > 0 && shown(el)) {
    let clipped = false;
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
      const ov = getComputedStyle(a).overflowX;
      if (ov !== 'visible' && a.getBoundingClientRect().right <= vw + 1) { clipped = true; break; }
    }
    if (!clipped) wide.push({el: describe(el), right: Math.round(r.right)});
  }
}

return {
  viewport: {w: vw, h: vh}, dark: matchMedia('(prefers-color-scheme: dark)').matches,
  scroll_width: scrollWidth, overflowing: wide, page_bg: pageBg, overlaps, clipped,
  texts, focusables, images, headings, form_controls: formControls, spacing, probes, root_styles: rootStyles,
  focus_rules: {copied: rulesRead, unreadable: rulesUnreadable},
  lang: root.getAttribute('lang') || '', title: document.title,
};
