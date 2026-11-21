"use strict";

const byId = (id) => document.getElementById(id);
const galleryVideos = document.querySelectorAll(".scene-video");

function enableGallery(section, scenes) {
  const viewer = section.querySelector(".viewer");
  const scenePicker = section.querySelector(".scene-picker");
  const list = section.querySelector(".scene-list");
  const count = section.querySelector(".scene-count");
  const video = section.querySelector(".scene-video");
  const referenceViews = viewer.querySelector(".reference-views");
  const playButton = section.querySelector(".play-scene");
  const title = section.querySelector(".scene-title");
  const errorMessage = section.querySelector(".video-error");
  const videoLinks = section.querySelectorAll(".scene-video a, .video-error a");
  const viewSwitch = section.querySelector(".view-switch");
  const outputs = viewSwitch.querySelectorAll("button");
  const motionImage = viewer.querySelector(".motion-reference");
  const motionImageHeading = viewer.querySelector(".motion-reference-heading");
  const motionImageUnavailable = viewer.querySelector(".motion-reference-unavailable");
  const pathCanvas = viewer.querySelector(".trajectory-canvas");
  const pathUnavailable = viewer.querySelector(".trajectory-unavailable");
  const trajectory = pathCanvas && typeof window.createTrajectory === "function"
    ? window.createTrajectory(pathCanvas, video) : null;
  const sceneButtons = new Map();
  let currentScene;
  let currentOutput = "render";
  let imageRequest = 0;
  let videoRevision = 0;

  function updateTrajectory() {
    if (trajectory && !pathCanvas.hidden) trajectory.update();
  }

  function showMotion(scene) {
    if (!motionImage) return;
    const data = window.argsWorldMotion?.[scene.id];
    const isGeneration = section.id === "generation";
    const reference = data?.reference;
    const imageSource = reference || (isGeneration ? "assets/initial-views/" + scene.id + ".webp" : null);
    const request = ++imageRequest;
    motionImageHeading.textContent = reference ? "Reference image" : (isGeneration ? "Initial generated view" : "Reference image");
    motionImage.hidden = true;
    motionImageUnavailable.hidden = false;
    const unavailableText = isGeneration && !reference ? "Initial generated view unavailable." : "Reference image unavailable.";
    motionImageUnavailable.textContent = imageSource ? "Loading image…" : unavailableText;
    motionImage.removeAttribute("src");
    if (imageSource) {
      motionImage.alt = (reference ? "Reference image for " : "Initial generated view of ") + scene.title;
      const image = new Image();
      image.onload = () => {
        if (request !== imageRequest) return;
        motionImage.src = imageSource;
        motionImage.hidden = false;
        motionImageUnavailable.hidden = true;
      };
      image.onerror = () => {
        if (request === imageRequest) motionImageUnavailable.textContent = unavailableText;
      };
      image.src = imageSource;
    }

    const hasPath = Boolean(trajectory && Array.isArray(data?.positions) && data.positions.length > 1);
    pathCanvas.hidden = !hasPath;
    pathUnavailable.hidden = hasPath;
    if (hasPath) {
      trajectory.setScene(data);
      updateTrajectory();
      requestAnimationFrame(updateTrajectory);
    }
  }

  function playVideo() {
    const revision = videoRevision;
    video.play().catch(() => {
      if (revision === videoRevision && video.paused && !video.error) playButton.hidden = false;
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
    if (currentScene !== scene) {
      showReferenceViews(scene.inputs ?? []);
      showMotion(scene);
    }
    currentScene = scene;
    const direct = scene.direct !== false && currentOutput === "direct";
    const base = "assets/gallery/" + scene.id + (direct ? "-direct" : "");
    const source = base + ".mp4";

    videoRevision++;
    video.pause();
    video.controls = false;
    video.poster = base + ".webp";
    if (video.getAttribute("src") !== source) video.src = source;
    video.setAttribute("aria-label", scene.title + (direct ? " direct RGB video" : " 3D Gaussian render"));
    for (const link of videoLinks) link.href = source;
    title.textContent = scene.title;
    viewSwitch.hidden = scene.direct === false;
    for (const [sceneId, button] of sceneButtons) {
      button.setAttribute("aria-pressed", String(sceneId === scene.id));
    }
    playButton.setAttribute("aria-label", "Play " + scene.title);
    playButton.hidden = false;
    errorMessage.hidden = true;

    if (continuePlayback) playVideo();
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
    list.append(button);
    sceneButtons.set(scene.id, button);
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
  for (const event of ["timeupdate", "seeking", "seeked", "loadedmetadata", "durationchange", "ended", "pause"]) {
    video.addEventListener(event, updateTrajectory);
  }
  video.addEventListener("error", () => {
    if (!video.error) return;
    playButton.hidden = true;
    errorMessage.hidden = false;
  });
  playButton.addEventListener("click", playVideo);

  count.textContent = scenes.length + (scenes.length === 1 ? " scene" : " scenes");
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
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let heroVisible = !("IntersectionObserver" in window);

function updateHeroPlayback() {
  if (heroVisible && !document.hidden && !reducedMotion.matches) {
    heroVideo.play().catch(() => {});
  } else {
    heroVideo.pause();
  }
}

reducedMotion.addEventListener("change", updateHeroPlayback);
document.addEventListener("visibilitychange", updateHeroPlayback);
heroVideo.muted = true;
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
