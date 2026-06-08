/**
 * Lavoisier Service Worker.
 *
 * Strategy:
 *   - Cache-first for the app shell (HTML, JS, CSS, fonts)
 *   - Cache-first for shaders (they never change without a deploy)
 *   - Network-first for external repository data (always fresh)
 *   - Bypass entirely for File System Access API operations
 *
 * The cache versions are bumped via the CACHE_VERSION constant on each
 * deploy. Old caches are pruned on activation.
 */

const CACHE_VERSION = "v1.1.0";
const SHELL_CACHE = `lavoisier-shell-${CACHE_VERSION}`;
const SHADER_CACHE = `lavoisier-shaders-${CACHE_VERSION}`;
const RUNTIME_CACHE = `lavoisier-runtime-${CACHE_VERSION}`;

const SHELL_ASSETS = [
  "/",
  "/tool",
  "/experiment",
  "/sandbox",
  "/framework",
  "/papers",
  "/about",
  "/manifest.webmanifest",
  "/favicon.ico",
];

const SHADER_ASSETS = [
  "/shaders/wave.vert",
  "/shaders/wave.frag",
  "/shaders/physics_overlay.frag",
  "/shaders/bijective_validation.frag",
  "/shaders/interference.frag",
  "/shaders/quality.frag",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    Promise.all([
      caches.open(SHELL_CACHE).then((cache) =>
        cache.addAll(SHELL_ASSETS).catch((err) => {
          console.warn("[sw] shell cache partial:", err);
        })
      ),
      caches.open(SHADER_CACHE).then((cache) =>
        cache.addAll(SHADER_ASSETS).catch((err) => {
          console.warn("[sw] shader cache partial:", err);
        })
      ),
    ]).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter(
            (key) =>
              key.startsWith("lavoisier-") &&
              ![SHELL_CACHE, SHADER_CACHE, RUNTIME_CACHE].includes(key)
          )
          .map((key) => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Bypass: same-origin POST/PUT, blob: URLs, chrome-extension://
  if (url.protocol !== "http:" && url.protocol !== "https:") return;

  // External repository data — always network-first
  if (url.host.includes("ebi.ac.uk") ||
      url.host.includes("zenodo.org") ||
      url.host.includes("massive.ucsd.edu") ||
      url.host.includes("ucsd.edu")) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Same origin: split by path
  if (url.origin === self.location.origin) {
    if (url.pathname.startsWith("/shaders/")) {
      event.respondWith(cacheFirst(request, SHADER_CACHE));
    } else if (
      url.pathname === "/" ||
      url.pathname === "/tool" ||
      url.pathname === "/experiment" ||
      url.pathname === "/sandbox" ||
      url.pathname === "/about" ||
      url.pathname === "/framework" ||
      url.pathname === "/papers"
    ) {
      // HTML routes: network-first so a deploy is picked up immediately,
      // falling back to cache offline.
      event.respondWith(networkFirst(request));
    } else if (url.pathname.startsWith("/_next/")) {
      // Hashed build assets are immutable per-deploy — safe to cache-first.
      event.respondWith(cacheFirst(request, SHELL_CACHE));
    } else {
      event.respondWith(networkFirst(request));
    }
  } else {
    // Other cross-origin requests — pass through
    event.respondWith(fetch(request).catch(() => caches.match(request)));
  }
});

async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone()).catch(() => {});
    }
    return response;
  } catch (err) {
    return caches.match(request);
  }
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(RUNTIME_CACHE);
      cache.put(request, response.clone()).catch(() => {});
    }
    return response;
  } catch (err) {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw err;
  }
}

// Allow the page to nudge the SW (e.g. "skipWaiting" after a deploy)
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
