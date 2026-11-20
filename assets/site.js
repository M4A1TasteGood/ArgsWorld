"use strict";

const byId = (id) => document.getElementById(id);
const galleryVideos = document.querySelectorAll(".scene-video");

function enableGallery(section, scenes) {
  const viewer = section.querySelector(".viewer");
  const scenePicker = section.querySelector(".scene-picker");
  const groups = section.querySelector(".scene-groups");
  const filters = section.querySelector(".scene-filters");
  const count = section.querySelector(".scene-count");
  const video = section.querySelector(".scene-video");
  const referenceViews = viewer.querySelector(".reference-views");
  const playButton = section.querySelector(".play-scene");
  const title = section.querySelector(".scene-title");
  const categoryLabel = section.querySelector(".scene-category");
  const errorMessage = section.querySelector(".video-error");
  const videoLinks = section.querySelectorAll(".scene-video a, .video-error a");
  const viewSwitch = section.querySelector(".view-switch");
  const outputs = viewSwitch.querySelectorAll("button");
  const categories = new Map();
  const categoryButtons = new Map();
  const sceneButtons = new Map();
  let currentScene;
  let currentOutput = "render";

  function playVideo() {
    video.play().catch(() => {
      if (video.paused && !video.error) playButton.hidden = false;
    });
  }

  function showReferenceViews(inputs) {
    if (!referenceViews) return;

    const hasInputs = inputs.length > 0;
    const list = referenceViews.querySelector(".reference-list");
    const images = inputs.map(({ slot, src }) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = src;
      link.target = "_blank";
      link.rel = "noopener";
      link.setAttribute("aria-label", "Open input image from camera " + slot + " in a new tab");

      const image = document.createElement("img");
      image.src = src;
      image.alt = "";
      image.width = 704;
      image.height = 480;
      image.decoding = "async";

      const label = document.createElement("span");
      label.textContent = "Camera " + slot;
      link.append(image, label);
      item.append(link);
      return item;
    });

    list.replaceChildren(...images);
    referenceViews.querySelector(".reference-count").textContent = inputs.length + (inputs.length === 1 ? " posed view" : " posed views");
    referenceViews.hidden = !hasInputs;
    viewer.classList.toggle("viewer--with-inputs", hasInputs);
  }

  function showScene(scene, continuePlayback = false) {
    if (currentScene !== scene) showReferenceViews(scene.inputs ?? []);
    currentScene = scene;
    const direct = scene.direct !== false && currentOutput === "direct";
    const base = "assets/gallery/" + scene.id + (direct ? "-direct" : "");
    const source = base + ".mp4";

    video.pause();
    video.controls = false;
    video.poster = base + ".webp";
    if (video.getAttribute("src") !== source) video.src = source;
    video.setAttribute("aria-label", scene.title + (direct ? " direct RGB video" : " 3D Gaussian render"));
    for (const link of videoLinks) link.href = source;
    title.textContent = scene.title;
    categoryLabel.textContent = scene.category;
    viewSwitch.hidden = scene.direct === false;
    for (const [sceneId, button] of sceneButtons) {
      button.setAttribute("aria-pressed", String(sceneId === scene.id));
    }
    playButton.setAttribute("aria-label", "Play " + scene.title);
    playButton.hidden = false;
    errorMessage.hidden = true;

    if (continuePlayback) playVideo();
  }

  function createSceneGroup(category) {
    const group = document.createElement("section");
    group.className = "scene-group";
    const heading = document.createElement("h3");
    heading.className = "scene-group-title";
    heading.id = section.id + "-category-" + categories.size;
    heading.textContent = category;
    group.setAttribute("aria-labelledby", heading.id);
    const list = document.createElement("div");
    list.className = "scene-list";
    group.append(heading, list);
    groups.append(group);
    return { group, list };
  }

  function selectCategory(category) {
    let visibleCount = 0;
    for (const [name, { group, list }] of categories) {
      const visible = category === null || category === name;
      group.hidden = !visible;
      if (visible) visibleCount += list.childElementCount;
    }
    for (const [name, filter] of categoryButtons) {
      filter.setAttribute("aria-pressed", String(category === name));
    }
    count.textContent = visibleCount + (visibleCount === 1 ? " scene" : " scenes");
  }

  for (const name of [...new Set(scenes.map((scene) => scene.category))].sort()) {
    categories.set(name, createSceneGroup(name));
  }

  for (const scene of scenes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-option";
    button.setAttribute("aria-controls", viewer.id);
    const thumbnail = document.createElement("span");
    thumbnail.className = "scene-thumbnail";

    const image = document.createElement("img");
    image.src = "assets/gallery/" + scene.id + "-thumb.webp";
    image.alt = "";
    image.width = 320;
    image.height = 218;
    image.decoding = "async";
    image.loading = "lazy";

    const playBadge = document.createElement("span");
    playBadge.className = "scene-thumbnail-play";
    playBadge.setAttribute("aria-hidden", "true");
    const playIcon = document.createElement("span");
    playIcon.className = "play-icon";
    playBadge.append(playIcon);
    thumbnail.append(image, playBadge);

    const name = document.createElement("strong");
    name.className = "scene-option-title";
    name.textContent = scene.title;
    button.append(thumbnail, name);
    button.addEventListener("click", () => {
      if (currentScene !== scene) showScene(scene);
      viewer.scrollIntoView({ block: "nearest", behavior: "instant" });
      const playbackControl = video.controls ? video : playButton;
      playbackControl.focus({ preventScroll: true });
      if (video.paused) playVideo();
    });
    categories.get(scene.category).list.append(button);
    sceneButtons.set(scene.id, button);
  }

  for (const category of [null, ...categories.keys()]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-filter";
    button.textContent = category ?? "All";
    button.addEventListener("click", () => selectCategory(category));
    filters.append(button);
    categoryButtons.set(category, button);
  }

  for (const button of outputs) {
    button.addEventListener("click", () => {
      if (button.dataset.output === currentOutput) return;
      currentOutput = button.dataset.output;
      for (const option of outputs) option.setAttribute("aria-pressed", String(option === button));
      showScene(currentScene, !video.paused);
    });
  }

  video.addEventListener("play", () => {
    for (const otherVideo of galleryVideos) {
      if (otherVideo !== video) otherVideo.pause();
    }
  });
  video.addEventListener("playing", () => {
    video.controls = true;
    if (document.activeElement === playButton) video.focus({ preventScroll: true });
    playButton.hidden = true;
  });
  video.addEventListener("error", () => {
    playButton.hidden = true;
    errorMessage.hidden = false;
  });
  playButton.addEventListener("click", playVideo);

  selectCategory(null);
  showScene(scenes[0]);
  scenePicker.hidden = false;
  viewer.querySelector(".browse-scenes").hidden = false;
}

for (const [id, scenes] of Object.entries(window.argsWorldScenes ?? {})) {
  if (scenes.length) enableGallery(byId(id), scenes);
}

if ("IntersectionObserver" in window) {
  const galleryObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) entry.target.pause();
    }
  });
  for (const video of galleryVideos) galleryObserver.observe(video);
}

const heroVideo = byId("hero-video");
const motionButton = byId("hero-motion");
const motionLabel = motionButton.querySelector(".motion-label");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let heroVisible = !("IntersectionObserver" in window);
let motionOverride;

function updateMotionControl() {
  const action = heroVideo.paused ? "Play" : "Pause";
  motionLabel.textContent = action + " video";
  motionButton.setAttribute("aria-label", action + " background video");
}

function updateHeroPlayback() {
  const wantsMotion = motionOverride ?? !reducedMotion.matches;
  if (heroVisible && !document.hidden && wantsMotion) {
    heroVideo.play().catch(updateMotionControl);
  } else {
    heroVideo.pause();
  }
  updateMotionControl();
}

motionButton.addEventListener("click", () => {
  motionOverride = heroVideo.paused;
  updateHeroPlayback();
});
heroVideo.addEventListener("play", updateMotionControl);
heroVideo.addEventListener("pause", updateMotionControl);
heroVideo.addEventListener("error", () => { motionButton.hidden = true; });
reducedMotion.addEventListener("change", updateHeroPlayback);
document.addEventListener("visibilitychange", updateHeroPlayback);
heroVideo.muted = true;
motionButton.hidden = false;
updateHeroPlayback();

if ("IntersectionObserver" in window) {
  const heroObserver = new IntersectionObserver(([entry]) => {
    heroVisible = entry.isIntersecting;
    updateHeroPlayback();
  });
  heroObserver.observe(heroVideo);
}

let copyReset;
byId("copy-citation").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const bibtex = byId("bibtex");
  try {
    await navigator.clipboard.writeText(bibtex.textContent);
    button.textContent = "Copied";
    byId("copy-status").textContent = "BibTeX copied to clipboard.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = "Text selected";
    byId("copy-status").textContent = "BibTeX selected for manual copying.";
  }
  clearTimeout(copyReset);
  copyReset = setTimeout(() => {
    button.textContent = "Copy BibTeX";
    byId("copy-status").textContent = "";
  }, 2400);
});
