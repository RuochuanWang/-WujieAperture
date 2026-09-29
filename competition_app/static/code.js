document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = button.dataset.copyTarget ? document.querySelector(`#${button.dataset.copyTarget}`) : null;
    const value = button.dataset.copy || target?.textContent || "";
    try {
      await navigator.clipboard.writeText(value.trim());
      const original = button.textContent;
      button.textContent = "已复制";
      window.setTimeout(() => { button.textContent = original; }, 1300);
    } catch (_) { button.textContent = "请手动复制"; }
  });
});

document.querySelectorAll("[data-deploy-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-deploy-tab]").forEach((item) => item.classList.toggle("is-active", item === button));
    document.querySelectorAll("[data-deploy-panel]").forEach((panel) => { panel.hidden = panel.dataset.deployPanel !== button.dataset.deployTab; });
  });
});
