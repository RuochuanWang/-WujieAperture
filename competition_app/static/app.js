const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const apertureManual = document.createElement("div");
apertureManual.className = "aperture-manual";
const apertureManualLabel = document.createElement("label");
apertureManualLabel.htmlFor = "apertureInput";
apertureManualLabel.textContent = "手动输入";
const apertureField = document.createElement("div");
apertureField.className = "aperture-field";
const aperturePrefix = document.createElement("span");
aperturePrefix.textContent = "f/";
aperturePrefix.setAttribute("aria-hidden", "true");
const apertureInput = document.createElement("input");
apertureInput.id = "apertureInput";
apertureInput.type = "number";
apertureInput.inputMode = "decimal";
apertureInput.min = "1.2";
apertureInput.max = "16";
apertureInput.step = "any";
apertureInput.value = "1.8";
apertureInput.setAttribute("aria-describedby", "apertureHint apertureError");
apertureField.append(aperturePrefix, apertureInput);
const apertureHint = document.createElement("small");
apertureHint.id = "apertureHint";
apertureHint.textContent = "VATD 可输入 1.2–16";
const apertureError = document.createElement("small");
apertureError.id = "apertureError";
apertureError.className = "field-error";
apertureError.setAttribute("role", "status");
apertureError.setAttribute("aria-live", "polite");
apertureError.hidden = true;
apertureManual.append(apertureManualLabel, apertureField, apertureHint, apertureError);
$("#apertureRange").insertAdjacentElement("beforebegin", apertureManual);

const elements = {
  fileInput: $("#fileInput"), browseButton: $("#browseButton"), replaceButton: $("#replaceButton"),
  dropZone: $("#dropZone"), previewStage: $("#previewStage"), imageFrame: $("#imageFrame"),
  sourceImage: $("#sourceImage"), depthImage: $("#depthImage"), depthLayer: $("#depthLayer"),
  resultImage: $("#resultImage"), resultLayer: $("#resultLayer"), compareLine: $("#compareLine"),
  focusMarker: $("#focusMarker"), focusTip: $("#focusTip"),
  depthBadge: $("#depthBadge"), afterBadge: $("#afterBadge"), apertureRange: $("#apertureRange"),
  apertureOutput: $("#apertureOutput"), apertureInput: $("#apertureInput"),
  apertureError: $("#apertureError"), engineHelp: $("#engineHelp"),
  sizeLimit: $("#sizeLimit"), customSizeGroup: $("#customSizeGroup"),
  customSizeInput: $("#customSizeInput"), sizeError: $("#sizeError"), sizeEstimate: $("#sizeEstimate"),
  renderButton: $("#renderButton"), downloadActions: $("#downloadActions"),
  downloadDepthButton: $("#downloadDepthButton"), downloadButton: $("#downloadButton"),
  resultMeta: $("#resultMeta"), processing: $("#processing"),
  processingTitle: $("#processingTitle"), processingDetail: $("#processingDetail"),
  errorToast: $("#errorToast"), errorMessage: $("#errorMessage"), fileMeta: $("#fileMeta"),
};

const state = {
  file: null,
  originalFile: null,
  sourceUrl: null,
  depthUrl: null,
  depthId: null,
  depthTime: 0,
  resultUrl: null,
  view: "source",
  focusPoint: null,
  compareValue: 52,
  comparePointerId: null,
  fNumber: 1.8,
  engine: "vatd",
  rendering: false,
  apertureValid: true,
  sizeValid: true,
  sizeDirty: false,
  preparedLongSide: null,
  loadToken: 0,
};
const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"]);
const maxBytes = 20 * 1024 * 1024;
const apertureMin = 1.2;
const apertureMax = 16;
const sizeMin = 256;
const sizeMax = 2048;

function sliderToAperture(value) {
  const t = Number(value) / 100;
  const inverse = (1 / apertureMin) * (1 - t) + (1 / apertureMax) * t;
  return Math.round((1 / inverse) * 10) / 10;
}

function apertureToSlider(value) {
  return Math.round(((1 / apertureMin - 1 / value) / (1 / apertureMin - 1 / apertureMax)) * 100);
}

function formatAperture(value) {
  return Number(Number(value).toFixed(2)).toString();
}

function showError(message) { elements.errorMessage.textContent = message; elements.errorToast.hidden = false; }
function dismissError() { elements.errorToast.hidden = true; }
function revokeUrl(name) { if (state[name]) URL.revokeObjectURL(state[name]); state[name] = null; }

function setView(mode) {
  if (mode === "depth" && !state.depthUrl) return;
  if (["result", "compare"].includes(mode) && !state.resultUrl) return;
  state.view = mode;
  $$('[data-view]').forEach((button) => {
    const selected = button.dataset.view === mode;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });

  const showDepth = mode === "depth";
  const showResult = mode === "result" || mode === "compare";
  const compare = mode === "compare";
  elements.depthLayer.hidden = !showDepth;
  elements.resultLayer.hidden = !showResult;
  elements.resultLayer.style.clipPath = compare
    ? `inset(0 ${100 - state.compareValue}% 0 0)`
    : "inset(0)";
  elements.compareLine.hidden = !compare;
  elements.sourceImage.style.visibility = mode === "result" ? "hidden" : "visible";
  $(".badge-before").hidden = !["source", "compare"].includes(mode);
  elements.depthBadge.hidden = !showDepth;
  elements.afterBadge.hidden = !showResult;
  syncInteractionUi();
}

function updateViewAvailability() {
  $('[data-view="depth"]').disabled = !state.depthUrl;
  $('[data-view="result"]').disabled = !state.resultUrl;
  $('[data-view="compare"]').disabled = !state.resultUrl;
}

function updateDownloadActions() {
  elements.downloadDepthButton.hidden = !state.depthUrl;
  elements.downloadButton.hidden = !state.resultUrl;
  elements.downloadActions.hidden = !state.depthUrl && !state.resultUrl;
}

function updateRenderLabel() {
  $("#renderButton span").textContent = !state.focusPoint
    ? "请先点击画面指定目标焦平面"
    : state.resultUrl ? "重新生成渲染结果" : "按当前焦平面生成渲染结果";
}

function clearResult() {
  revokeUrl("resultUrl");
  elements.resultImage.removeAttribute("src");
  elements.resultMeta.hidden = true;
  updateViewAvailability();
  updateDownloadActions();
  updateRenderLabel();
  if (["result", "compare"].includes(state.view)) setView(state.depthUrl ? "depth" : "source");
}

function clearDepth() {
  clearResult();
  revokeUrl("depthUrl");
  state.depthId = null;
  state.depthTime = 0;
  elements.depthImage.removeAttribute("src");
  updateViewAvailability();
  updateDownloadActions();
  updateRenderLabel();
  if (state.view === "depth") setView("source");
}

function updateApertureUi(syncInput = true) {
  const formatted = formatAperture(state.fNumber);
  elements.apertureOutput.textContent = `f/${formatted}`;
  if (syncInput) elements.apertureInput.value = formatted;
  elements.apertureRange.style.setProperty("--range-progress", `${elements.apertureRange.value}%`);
  $$('[data-f]').forEach((button) => button.classList.toggle("is-active", Math.abs(Number(button.dataset.f) - state.fNumber) < .06));
}

function setApertureError(message = "") {
  state.apertureValid = !message;
  elements.apertureInput.setCustomValidity(message);
  elements.apertureInput.setAttribute("aria-invalid", String(Boolean(message)));
  elements.apertureError.textContent = message;
  elements.apertureError.hidden = !message;
  syncInteractionUi();
}

function readManualAperture() {
  if (state.resultUrl) clearResult();
  const raw = elements.apertureInput.value.trim();
  const value = Number(raw);
  if (!raw) return setApertureError("请输入光圈值。");
  if (!Number.isFinite(value)) return setApertureError("请输入有效的光圈数值。");
  if (value < apertureMin) return setApertureError("光圈不能小于 f/1.2。");
  if (value > apertureMax) return setApertureError("光圈不能大于 f/16。");
  setApertureError();
  state.fNumber = value;
  elements.apertureRange.value = apertureToSlider(value);
  updateApertureUi(false);
  if (state.resultUrl) clearResult();
}

function setAperture(value) {
  state.fNumber = Number(value);
  setApertureError();
  elements.apertureRange.value = apertureToSlider(state.fNumber);
  updateApertureUi();
  if (state.resultUrl) clearResult();
}

function setEngine(engine) {
  state.engine = engine;
  $$('[data-engine]').forEach((button) => {
    const selected = button.dataset.engine === engine;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-checked", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  const ebb = engine === "ebb";
  if (ebb) setAperture(1.8);
  elements.apertureRange.disabled = ebb || state.rendering;
  elements.apertureInput.disabled = ebb || state.rendering;
  $$('[data-f]').forEach((button) => { button.disabled = ebb || state.rendering; });
  elements.engineHelp.textContent = ebb
    ? "读取 EBB 的权重参数，光圈固定为 f/1.8。"
    : "读取 VATD 的权重参数，支持 f/1.2–f/16。";
  if (state.resultUrl) clearResult();
}

function syncInteractionUi() {
  const canPickFocus = Boolean(state.file) && !state.rendering && state.sizeValid && !state.sizeDirty && state.view !== "compare";
  elements.imageFrame.classList.toggle("is-focusable", canPickFocus);
  elements.imageFrame.classList.toggle("is-comparing", state.view === "compare");
  elements.focusTip.hidden = !canPickFocus || Boolean(state.focusPoint);
  elements.focusMarker.hidden = !state.focusPoint || state.view === "compare";
  elements.renderButton.disabled = state.rendering || !state.apertureValid || !state.sizeValid || state.sizeDirty || !state.file || !state.focusPoint;
  updateRenderLabel();
}

function readProcessingLongSide() {
  const custom = elements.sizeLimit.value === "custom";
  const raw = custom ? elements.customSizeInput.value.trim() : elements.sizeLimit.value;
  if (!raw) return { value: null, error: "请输入处理尺寸。" };
  const value = Number(raw);
  if (!Number.isInteger(value)) return { value: null, error: "处理尺寸须为整数像素。" };
  if (value < sizeMin) return { value: null, error: `最长边不能小于 ${sizeMin} px。` };
  if (value > sizeMax) return { value: null, error: `最长边不能超过 ${sizeMax} px。` };
  if (value % 4 !== 0) return { value: null, error: "处理尺寸需为 4 的倍数。" };
  return { value, error: "" };
}

function setSizeError(message = "") {
  state.sizeValid = !message;
  elements.customSizeInput.setCustomValidity(message);
  elements.customSizeInput.setAttribute("aria-invalid", String(Boolean(message)));
  elements.sizeError.textContent = message;
  elements.sizeError.hidden = !message;
  syncInteractionUi();
}

function updateSizeUi() {
  const custom = elements.sizeLimit.value === "custom";
  elements.customSizeGroup.hidden = !custom;
  const { value, error } = readProcessingLongSide();
  setSizeError(error);
  if (error) {
    elements.sizeEstimate.textContent = "自定义尺寸无效";
    state.sizeDirty = true;
    syncInteractionUi();
    return null;
  }
  elements.sizeEstimate.textContent = `最长边 ${value} px`;
  state.sizeDirty = Boolean(state.file && state.preparedLongSide !== value);
  syncInteractionUi();
  return value;
}

function canvasToBlob(canvas, type) {
  return new Promise((resolve, reject) => canvas.toBlob(
    (blob) => blob ? resolve(blob) : reject(new Error("浏览器无法压缩这张图片。")),
    type,
    0.92,
  ));
}

async function prepareFile(file, maxLongSide) {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const originalWidth = bitmap.width;
  const originalHeight = bitmap.height;
  if (originalWidth < 64 || originalHeight < 64) {
    bitmap.close();
    throw new Error("图片尺寸过小，宽高至少为 64 像素。");
  }
  const scale = Math.min(1, maxLongSide / Math.max(originalWidth, originalHeight));
  const width = Math.max(64, Math.round(originalWidth * scale));
  const height = Math.max(64, Math.round(originalHeight * scale));
  if (scale === 1) {
    bitmap.close();
    return { file, originalWidth, originalHeight, width, height, compressed: false };
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const type = file.type === "image/jpeg" ? "image/jpeg" : "image/webp";
  const blob = await canvasToBlob(canvas, type);
  const processed = new File([blob], file.name, { type, lastModified: file.lastModified });
  return { file: processed, originalWidth, originalHeight, width, height, compressed: true };
}

async function loadFile(file, rememberOriginal = true) {
  dismissError();
  if (!file || !allowedTypes.has(file.type)) return showError("请选择 JPG、PNG、WebP、BMP 或 TIFF 图片。");
  if (file.size > maxBytes) return showError("图片不能超过 20 MB，请压缩后再试。");
  const requestedLongSide = updateSizeUi();
  if (!requestedLongSide) return showError("请先输入有效的处理尺寸。");
  if (rememberOriginal) state.originalFile = file;
  const token = ++state.loadToken;
  state.sizeDirty = true;
  syncInteractionUi();
  elements.fileMeta.textContent = "正在按所选分辨率准备图像…";
  elements.sizeLimit.disabled = true;
  elements.customSizeInput.disabled = true;
  let prepared;
  try {
    prepared = await prepareFile(file, requestedLongSide);
  } catch (error) {
    if (token === state.loadToken) showError(error.message || "无法读取或压缩这张图片。");
    elements.sizeLimit.disabled = false;
    elements.customSizeInput.disabled = false;
    state.sizeDirty = false;
    syncInteractionUi();
    return;
  }
  if (token !== state.loadToken) return;
  state.file = prepared.file;
  state.preparedLongSide = requestedLongSide;
  state.sizeDirty = false;
  state.focusPoint = null;
  revokeUrl("sourceUrl");
  clearDepth();
  state.sourceUrl = URL.createObjectURL(prepared.file);
  elements.sourceImage.src = state.sourceUrl;
  const dimensions = prepared.compressed
    ? `${prepared.originalWidth}×${prepared.originalHeight} → ${prepared.width}×${prepared.height}`
    : `${prepared.width}×${prepared.height} · 无需压缩`;
  const megapixels = (prepared.width * prepared.height / 1_000_000).toFixed(2);
  elements.fileMeta.textContent = `${file.name} · ${dimensions} · ${megapixels} MP`;
  elements.sizeEstimate.textContent = `处理 ${prepared.width} × ${prepared.height}`;
  elements.dropZone.hidden = true;
  elements.previewStage.hidden = false;
  elements.replaceButton.hidden = false;
  elements.renderButton.disabled = true;
  elements.focusMarker.hidden = true;
  elements.fileInput.value = "";
  elements.sizeLimit.disabled = false;
  elements.customSizeInput.disabled = false;
  setView("source");
}

function updateCompare(value) {
  state.compareValue = Math.min(100, Math.max(0, Number(value)));
  elements.resultLayer.style.clipPath = `inset(0 ${100 - state.compareValue}% 0 0)`;
  elements.compareLine.style.left = `${state.compareValue}%`;
  elements.compareLine.setAttribute("aria-valuenow", String(Math.round(state.compareValue)));
}

function pickFocus(event) {
  if (!state.file || state.rendering || state.view === "compare") return;
  const rect = elements.imageFrame.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  const y = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
  state.focusPoint = [x, y];
  elements.focusMarker.style.left = `${x * 100}%`;
  elements.focusMarker.style.top = `${y * 100}%`;
  clearResult();
  syncInteractionUi();
  render();
}

function updateCompareFromPointer(event) {
  const rect = elements.imageFrame.getBoundingClientRect();
  updateCompare(((event.clientX - rect.left) / rect.width) * 100);
}

function startCompare(event) {
  if (state.view !== "compare" || state.rendering || event.button !== 0) return;
  state.comparePointerId = event.pointerId;
  elements.imageFrame.setPointerCapture(event.pointerId);
  updateCompareFromPointer(event);
  event.preventDefault();
}

function moveCompare(event) {
  if (state.comparePointerId !== event.pointerId) return;
  updateCompareFromPointer(event);
}

function stopCompare(event) {
  if (state.comparePointerId !== event.pointerId) return;
  state.comparePointerId = null;
  if (elements.imageFrame.hasPointerCapture(event.pointerId)) {
    elements.imageFrame.releasePointerCapture(event.pointerId);
  }
}

function adjustCompareWithKeyboard(event) {
  if (state.view !== "compare") return;
  const values = { ArrowLeft: state.compareValue - 2, ArrowRight: state.compareValue + 2, Home: 0, End: 100 };
  if (!(event.key in values)) return;
  updateCompare(values[event.key]);
  event.preventDefault();
}

function setProcessing(active) {
  state.rendering = active;
  elements.processing.hidden = !active;
  elements.processing.classList.toggle("has-depth", active && Boolean(state.depthUrl));
  elements.renderButton.disabled = active || !state.apertureValid || !state.sizeValid || state.sizeDirty || !state.file || !state.focusPoint;
  elements.replaceButton.disabled = active;
  elements.apertureRange.disabled = active || state.engine === "ebb";
  elements.apertureInput.disabled = active || state.engine === "ebb";
  elements.sizeLimit.disabled = active;
  elements.customSizeInput.disabled = active;
  $$('[data-f]').forEach((button) => { button.disabled = active || state.engine === "ebb"; });
  $$('[data-engine]').forEach((button) => { button.disabled = active; });
  syncInteractionUi();
}

function setProcessingPhase(index) {
  const phases = [
    ["正在生成深度图", "DepthPro 正在分析图像空间结构"],
    ["正在生成渲染结果", "正在计算渲染结果"],
  ];
  elements.processingTitle.textContent = phases[index][0];
  elements.processingDetail.textContent = phases[index][1];
  $$(".processing-steps i").forEach((step, i) => step.classList.toggle("is-active", i <= index));
}

async function apiError(response) {
  try {
    const body = await response.json();
    return { code: body?.error?.code || "HTTP_ERROR", message: body?.error?.message || `生成失败（HTTP ${response.status}）` };
  } catch (_) {
    return { code: "HTTP_ERROR", message: `生成失败（HTTP ${response.status}）` };
  }
}

async function estimateDepth() {
  const body = new FormData();
  body.append("image", state.file);
  body.append("max_long_side", String(state.preparedLongSide));
  const response = await fetch("/api/depth", { method: "POST", body });
  if (!response.ok) throw await apiError(response);
  const depthId = response.headers.get("X-Wujie-Aperture-Depth-Id");
  if (!depthId) throw { code: "INVALID_DEPTH_RESPONSE", message: "服务未返回深度标识，请检查后端版本。" };

  revokeUrl("depthUrl");
  state.depthUrl = URL.createObjectURL(await response.blob());
  state.depthId = depthId;
  state.depthTime = Number(response.headers.get("X-Wujie-Aperture-Depth-Time") || 0);
  elements.depthImage.src = state.depthUrl;
  updateViewAvailability();
  updateDownloadActions();
  updateRenderLabel();
  elements.processing.classList.add("has-depth");
}

async function requestBokeh() {
  const body = new FormData();
  body.append("image", state.file);
  body.append("engine", state.engine);
  body.append("f_number", String(state.fNumber));
  body.append("depth_id", state.depthId);
  body.append("focus_x", String(state.focusPoint[0]));
  body.append("focus_y", String(state.focusPoint[1]));
  body.append("max_long_side", String(state.preparedLongSide));
  return fetch("/api/render", { method: "POST", body });
}

async function render() {
  if (!state.file || state.rendering) return;
  if (!state.focusPoint) return showError("请先点击图像指定目标焦平面。");
  if (!state.apertureValid) {
    elements.apertureInput.focus();
    return;
  }
  if (!state.sizeValid || state.sizeDirty || !state.preparedLongSide) {
    updateSizeUi();
    return showError("处理尺寸已更改，请等待图像按新分辨率准备完成后再渲染。");
  }
  dismissError();
  clearResult();
  setProcessing(true);
  setProcessingPhase(state.depthId ? 1 : 0);
  try {
    if (!state.depthId) await estimateDepth();
    setProcessingPhase(1);
    let response = await requestBokeh();
    if (!response.ok) {
      const error = await apiError(response);
      if (error.code !== "DEPTH_RESULT_NOT_FOUND") throw error;
      clearDepth();
      setProcessingPhase(0);
      await estimateDepth();
      setProcessingPhase(1);
      response = await requestBokeh();
      if (!response.ok) throw await apiError(response);
    }

    state.resultUrl = URL.createObjectURL(await response.blob());
    elements.resultImage.src = state.resultUrl;
    updateViewAvailability();
    updateDownloadActions();
    updateCompare(state.compareValue);
    setView("result");
    const engine = response.headers.get("X-Wujie-Aperture-Engine") || state.engine.toUpperCase();
    const renderTime = Number(response.headers.get("X-Wujie-Aperture-Render-Time") || 0);
    elements.resultMeta.textContent = `渲染输出 · ${engine} · ${response.headers.get("X-Wujie-Aperture-Width")}×${response.headers.get("X-Wujie-Aperture-Height")} · f/${formatAperture(state.fNumber)} · 深度 ${state.depthTime.toFixed(1)}s / 渲染 ${renderTime.toFixed(1)}s`;
    elements.resultMeta.hidden = false;
  } catch (error) {
    showError(error.message || "生成失败，请检查模型配置后重试。");
  } finally {
    setProcessing(false);
  }
}

function download(url, suffix) {
  if (!url) return;
  const link = document.createElement("a");
  const stem = state.file.name.replace(/\.[^.]+$/, "");
  link.href = url;
  link.download = `${stem}_${suffix}.png`;
  link.click();
}

elements.browseButton.addEventListener("click", (event) => { event.stopPropagation(); elements.fileInput.click(); });
elements.replaceButton.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", (event) => loadFile(event.target.files[0]));
elements.dropZone.addEventListener("click", () => elements.fileInput.click());
["dragenter", "dragover"].forEach((type) => elements.dropZone.addEventListener(type, (event) => { event.preventDefault(); elements.dropZone.classList.add("is-dragging"); }));
["dragleave", "drop"].forEach((type) => elements.dropZone.addEventListener(type, (event) => { event.preventDefault(); elements.dropZone.classList.remove("is-dragging"); }));
elements.dropZone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));
elements.sizeLimit.addEventListener("change", async () => {
  const target = updateSizeUi();
  if (elements.sizeLimit.value === "custom") elements.customSizeInput.focus();
  if (target && state.originalFile) await loadFile(state.originalFile, false);
});
elements.customSizeInput.addEventListener("input", () => {
  updateSizeUi();
  if (state.file && state.sizeDirty) clearDepth();
});
elements.customSizeInput.addEventListener("change", async () => {
  const target = updateSizeUi();
  if (target && state.originalFile) await loadFile(state.originalFile, false);
});
elements.customSizeInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && state.sizeValid) elements.customSizeInput.blur();
});
elements.apertureRange.addEventListener("input", () => { state.fNumber = sliderToAperture(elements.apertureRange.value); setApertureError(); updateApertureUi(); if (state.resultUrl) clearResult(); });
elements.apertureInput.addEventListener("input", readManualAperture);
elements.apertureInput.addEventListener("blur", () => {
  if (state.apertureValid) elements.apertureInput.value = formatAperture(state.fNumber);
});
elements.apertureInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && state.apertureValid) elements.apertureInput.blur();
});
$$('[data-f]').forEach((button) => button.addEventListener("click", () => setAperture(button.dataset.f)));
const engineButtons = $$('[data-engine]');
engineButtons.forEach((button) => {
  button.addEventListener("click", () => setEngine(button.dataset.engine));
  button.addEventListener("keydown", (event) => {
    if (button.disabled) return;
    const current = engineButtons.indexOf(button);
    const target = ({
      ArrowLeft: (current - 1 + engineButtons.length) % engineButtons.length,
      ArrowUp: (current - 1 + engineButtons.length) % engineButtons.length,
      ArrowRight: (current + 1) % engineButtons.length,
      ArrowDown: (current + 1) % engineButtons.length,
      Home: 0,
      End: engineButtons.length - 1,
    })[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const next = engineButtons[target];
    setEngine(next.dataset.engine);
    next.focus();
  });
});
elements.imageFrame.addEventListener("click", pickFocus);
elements.imageFrame.addEventListener("keydown", (event) => {
  if (event.target !== elements.imageFrame || !state.file || state.rendering || state.view === "compare") return;
  if (event.key === "Enter") {
    if (!state.focusPoint) state.focusPoint = [0.5, 0.5];
    elements.focusMarker.style.left = `${state.focusPoint[0] * 100}%`;
    elements.focusMarker.style.top = `${state.focusPoint[1] * 100}%`;
    syncInteractionUi();
    render();
    event.preventDefault();
    return;
  }
  const steps = { ArrowLeft: [-0.02, 0], ArrowRight: [0.02, 0], ArrowUp: [0, -0.02], ArrowDown: [0, 0.02] };
  if (!(event.key in steps)) return;
  const point = state.focusPoint || [0.5, 0.5];
  state.focusPoint = point.map((value, i) => Math.max(0, Math.min(1, value + steps[event.key][i])));
  elements.focusMarker.style.left = `${state.focusPoint[0] * 100}%`;
  elements.focusMarker.style.top = `${state.focusPoint[1] * 100}%`;
  clearResult();
  syncInteractionUi();
  event.preventDefault();
});
elements.imageFrame.addEventListener("pointerdown", startCompare);
elements.imageFrame.addEventListener("pointermove", moveCompare);
elements.imageFrame.addEventListener("pointerup", stopCompare);
elements.imageFrame.addEventListener("pointercancel", stopCompare);
elements.compareLine.addEventListener("keydown", adjustCompareWithKeyboard);
$$('[data-view]').forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
elements.renderButton.addEventListener("click", render);
elements.downloadDepthButton.addEventListener("click", () => download(state.depthUrl, "depthpro_depth"));
elements.downloadButton.addEventListener("click", () => download(state.resultUrl, `wujie_aperture_physical_${state.engine}_f${formatAperture(state.fNumber)}`));
$("#dismissError").addEventListener("click", dismissError);
window.addEventListener("beforeunload", () => { revokeUrl("sourceUrl"); revokeUrl("depthUrl"); revokeUrl("resultUrl"); });

updateSizeUi();
setAperture(1.8);
setEngine("vatd");
setView("source");
updateViewAvailability();
updateDownloadActions();
syncInteractionUi();
