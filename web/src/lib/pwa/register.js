/**
 * Service worker registration.
 *
 * Registers /sw.js once the page is idle. Skips registration in
 * development to avoid stale-cache surprises.
 */

export function registerServiceWorker() {
  if (typeof window === "undefined") return;
  if (!("serviceWorker" in navigator)) return;

  if (window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1") {
    return; // skip in dev
  }

  const register = async () => {
    try {
      const reg = await navigator.serviceWorker.register("/sw.js", { scope: "/" });

      // Always check for a newer SW on load.
      reg.update().catch(() => {});

      // When a new service worker finishes installing while an old one is
      // controlling the page, tell it to activate immediately, then reload
      // once it takes control so the user always sees the latest deploy.
      reg.addEventListener("updatefound", () => {
        const newWorker = reg.installing;
        if (!newWorker) return;
        newWorker.addEventListener("statechange", () => {
          if (newWorker.state === "installed" && navigator.serviceWorker.controller) {
            newWorker.postMessage({ type: "SKIP_WAITING" });
          }
        });
      });

      // Reload exactly once when the new SW takes control.
      let refreshing = false;
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (refreshing) return;
        refreshing = true;
        window.location.reload();
      });
    } catch (err) {
      console.warn("[lavoisier] service worker registration failed:", err);
    }
  };

  if (document.readyState === "complete") {
    register();
  } else {
    window.addEventListener("load", register, { once: true });
  }
}

/**
 * Returns true if the app is running standalone (installed as PWA).
 */
export function isStandalone() {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia?.("(display-mode: standalone)").matches ||
    window.navigator?.standalone === true
  );
}
