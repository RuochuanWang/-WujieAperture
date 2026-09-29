import fs from "node:fs";

const debugPort = process.argv[2] || "9222";
const inputPath = process.argv[3];
const screenshotPath = process.argv[4];

if (!inputPath || !screenshotPath) {
  throw new Error("Usage: node validate_browser_interaction.mjs PORT INPUT SCREENSHOT");
}

const targets = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then((response) => response.json());
const target = targets.find((item) => item.type === "page");
if (!target) throw new Error("No Chrome page target found");

const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let commandId = 0;
const pending = new Map();
const browserErrors = [];
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.error) request.reject(new Error(message.error.message));
    else request.resolve(message.result);
    return;
  }
  if (message.method === "Runtime.exceptionThrown") {
    browserErrors.push(message.params.exceptionDetails.text);
  }
});

function send(method, params = {}) {
  const id = ++commandId;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

async function evaluate(expression, returnByValue = true) {
  const response = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue });
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
  return response.result.value;
}

async function waitFor(expression, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Timed out waiting for: ${expression}`);
}

await send("Page.enable");
await send("Runtime.enable");
await send("DOM.enable");
await send("Page.navigate", { url: "http://127.0.0.1:7860/studio" });
await waitFor("document.readyState === 'complete'");

const documentNode = await send("DOM.getDocument");
const inputNode = await send("DOM.querySelector", {
  nodeId: documentNode.root.nodeId,
  selector: "#fileInput",
});
await send("DOM.setFileInputFiles", { nodeId: inputNode.nodeId, files: [inputPath] });
await evaluate("document.querySelector('#fileInput').dispatchEvent(new Event('change', { bubbles: true }))");
await waitFor("!document.querySelector('#previewStage').hidden && document.querySelector('#sourceImage').complete");

const focusRect = await evaluate(`(() => {
  const rect = document.querySelector('#imageFrame').getBoundingClientRect();
  return { x: rect.left + rect.width * 0.52, y: rect.top + rect.height * 0.48 };
})()`);
await send("Input.dispatchMouseEvent", { type: "mousePressed", x: focusRect.x, y: focusRect.y, button: "left", clickCount: 1 });
await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: focusRect.x, y: focusRect.y, button: "left", clickCount: 1 });

await waitFor(`document.querySelector('[data-view="result"]').classList.contains('is-active') &&
  document.querySelector('#processing').hidden &&
  document.querySelector('#resultImage').src.startsWith('blob:')`);

await evaluate("document.querySelector('[data-view=\"compare\"]').click()");
await waitFor("document.querySelector('[data-view=\"compare\"]').classList.contains('is-active')");
const compareRect = await evaluate(`(() => {
  const rect = document.querySelector('#imageFrame').getBoundingClientRect();
  return { left: rect.left, width: rect.width, y: rect.top + rect.height / 2 };
})()`);
const startX = compareRect.left + compareRect.width * 0.30;
const endX = compareRect.left + compareRect.width * 0.78;
await send("Input.dispatchMouseEvent", { type: "mousePressed", x: startX, y: compareRect.y, button: "left", buttons: 1, clickCount: 1 });
await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: endX, y: compareRect.y, button: "left", buttons: 1 });
await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: endX, y: compareRect.y, button: "left", buttons: 0, clickCount: 1 });

const result = await evaluate(`(() => ({
  view: document.querySelector('.canvas-tabs .is-active').dataset.view,
  lineLeft: document.querySelector('#compareLine').style.left,
  ariaValue: document.querySelector('#compareLine').getAttribute('aria-valuenow'),
  smartFocusVisible: document.body.textContent.includes('智能主体'),
  focusHeader: document.querySelector('#resultMeta').textContent,
}))()`);
if (Math.abs(Number(result.ariaValue) - 78) > 2) {
  throw new Error(`Compare drag did not reach 78%: ${JSON.stringify(result)}`);
}
if (result.smartFocusVisible) throw new Error("Smart focus control is still visible");
if (browserErrors.length) throw new Error(`Browser exceptions: ${browserErrors.join("; ")}`);

const screenshot = await send("Page.captureScreenshot", { format: "png" });
fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, "base64"));
process.stdout.write(`${JSON.stringify(result)}\n`);
await send("Browser.close");
socket.close();
