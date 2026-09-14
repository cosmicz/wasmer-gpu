"use strict";
const tabs = [...document.querySelectorAll('[role="tab"]')];

function selectTab(name, focus = false) {
  const selected = tabs.find(tab => tab.dataset.app === name) || tabs[1];
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    const panel = document.getElementById(tab.getAttribute("aria-controls"));
    panel.hidden = !active;
    const frame = panel.querySelector("iframe");
    // Load once. Switching tabs must never throw away a run or prediction.
    if (active && !frame.hasAttribute("src")) frame.src = frame.dataset.src;
  }
  if (focus) selected.focus();
  history.replaceState(null, "", `#${selected.dataset.app}`);
  document.title = `${selected.textContent} · Wasmer GPU`;
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab.dataset.app));
  tab.addEventListener("keydown", event => {
    const target = { ArrowLeft: (index + 2) % 3, ArrowRight: (index + 1) % 3, Home: 0, End: 2 }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    selectTab(tabs[target].dataset.app, true);
  });
});
window.addEventListener("hashchange", () => selectTab(location.hash.slice(1)));
selectTab(location.hash.slice(1));
