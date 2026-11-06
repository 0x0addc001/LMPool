(() => {
  const root = document.documentElement;
  const languageButton = document.querySelector(".language-toggle");
  const themeButton = document.querySelector(".theme-toggle");
  const translatable = document.querySelectorAll("[data-en][data-zh]");

  function readStorage(key, fallback) {
    try {
      return localStorage.getItem(key) || fallback;
    } catch {
      return fallback;
    }
  }

  function writeStorage(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      // The page remains functional when storage is unavailable.
    }
  }

  let language = readStorage("lmpool-language", "en");
  let theme = readStorage("lmpool-theme", "dark");

  function applyLanguage(next) {
    language = next === "zh" ? "zh" : "en";
    root.lang = language;
    root.dataset.language = language;
    translatable.forEach((element) => {
      element.textContent = element.dataset[language];
    });
    document.querySelectorAll(".lang-en").forEach((element) => element.classList.toggle("is-current", language === "en"));
    document.querySelectorAll(".lang-zh").forEach((element) => element.classList.toggle("is-current", language === "zh"));
    languageButton?.setAttribute("aria-label", language === "en" ? "Switch to Chinese" : "切换至英文");
    writeStorage("lmpool-language", language);
  }

  function applyTheme(next) {
    theme = next === "light" ? "light" : "dark";
    root.dataset.theme = theme;
    themeButton?.setAttribute("aria-pressed", String(theme === "light"));
    themeButton?.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#07111f" : "#f2f6fb");
    writeStorage("lmpool-theme", theme);
  }

  languageButton?.addEventListener("click", () => applyLanguage(language === "en" ? "zh" : "en"));
  themeButton?.addEventListener("click", () => applyTheme(theme === "dark" ? "light" : "dark"));
  applyLanguage(language);
  applyTheme(theme);
})();
