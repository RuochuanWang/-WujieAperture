const hero = document.querySelector(".home-hero");
const toggle = document.querySelector("#heroMotionToggle");
const frames = [...document.querySelectorAll(".hero-frame")];
const ticks = [...document.querySelectorAll(".scale-tick")];
const track = document.querySelector(".scale-track");
const cursor = document.querySelector("#apertureCursor");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const segmentMs = 1600;
const lastFrame = frames.length - 1;
const cycleMs = segmentMs * lastFrame * 2;
let trackWidth = 0;
let position = 0;
let timeOrigin = 0;
let pausedElapsed = 0;
let animationFrame = 0;
let ready = false;
let paused = false;
toggle.disabled = true;

function showPosition(nextPosition) {
  position = nextPosition;
  const lower = Math.floor(position);
  const blend = position - lower;
  frames.forEach((image, index) => {
    image.style.opacity = index === lower ? "1" : index === lower + 1 ? String(blend) : "0";
  });
  cursor.style.left = "-1px";
  cursor.style.transform = `translate3d(${trackWidth * (1 - position / lastFrame)}px, 0, 0)`;
  ticks.forEach(tick => tick.classList.toggle("is-near", Math.abs(Number(tick.dataset.frame) - position) < 0.08));
}

function paint(now) {
  const elapsed = (now - timeOrigin) % cycleMs;
  const steps = elapsed / segmentMs;
  showPosition(steps <= lastFrame ? steps : 2 * lastFrame - steps);
  animationFrame = requestAnimationFrame(paint);
}

function start() {
  if (!ready || paused || reducedMotion.matches) return;
  timeOrigin = performance.now() - pausedElapsed;
  animationFrame = requestAnimationFrame(paint);
}

toggle?.addEventListener("click", () => {
  paused = !paused;
  if (paused) {
    pausedElapsed = (performance.now() - timeOrigin) % cycleMs;
    cancelAnimationFrame(animationFrame);
  } else {
    start();
  }
  hero.classList.toggle("is-paused", paused);
  toggle.setAttribute("aria-pressed", String(paused));
  toggle.textContent = paused ? "继续演示" : "暂停演示";
});

function syncReducedMotion() {
  cancelAnimationFrame(animationFrame);
  if (reducedMotion.matches) {
    if (ready && !paused) pausedElapsed = (performance.now() - timeOrigin) % cycleMs;
    frames.forEach(image => image.style.removeProperty("opacity"));
    cursor.style.removeProperty("left");
    cursor.style.removeProperty("transform");
    ticks.forEach(tick => tick.classList.remove("is-near"));
    toggle.hidden = true;
  } else {
    toggle.hidden = false;
    if (ready) {
      showPosition(position);
      start();
    }
  }
}

reducedMotion.addEventListener("change", syncReducedMotion);

new ResizeObserver(() => {
  trackWidth = track.getBoundingClientRect().width;
  if (ready && !reducedMotion.matches) showPosition(position);
}).observe(track);

async function waitForImage(image) {
  if (!image.complete) {
    await new Promise((resolve, reject) => {
      image.addEventListener("load", resolve, { once: true });
      image.addEventListener("error", reject, { once: true });
    });
  }
  if (!image.naturalWidth) throw new Error("案例图片无法加载");
  try { await image.decode(); } catch { /* The loaded pixels remain usable. */ }
}

Promise.all(frames.map(waitForImage)).then(() => {
  ready = true;
  toggle.disabled = false;
  trackWidth = track.getBoundingClientRect().width;
  if (reducedMotion.matches) {
    toggle.hidden = true;
  } else {
    showPosition(0);
    start();
  }
}).catch(() => {
  hero.classList.add("demo-unavailable");
  const name = document.querySelector(".instrument-name span:last-child");
  if (name) name.textContent = "光圈序列暂时无法加载";
  toggle.hidden = true;
});
