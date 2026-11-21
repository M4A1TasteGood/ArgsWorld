/* Camera-path visualization. Positions are supplied by the scene, never inferred. */
(function () {
  "use strict";

  window.createTrajectory = function createTrajectory(canvas, video) {
    const ctx = canvas.getContext("2d");
    let positions = [];
    let starts = [];
    let sceneData = null;
    let activePositions = null;
    let frameRequest = 0;
    const colors = ["#4169db", "#e59446", "#30a08b", "#ac69bb", "#df7583", "#6695bd"];

    function isDark() {
      for (let node = canvas; node; node = node.parentElement) {
        const color = getComputedStyle(node).backgroundColor;
        const values = color.match(/[\d.]+/g)?.map(Number);
        if (values && values.length >= 3 && (values.length < 4 || values[3] > 0.5)) {
          return (values[0] * 0.299 + values[1] * 0.587 + values[2] * 0.114) < 140;
        }
      }
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }

    function roundedRect(x, y, width, height, radius) {
      radius = Math.max(0, Math.min(radius, width / 2, height / 2));
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.arcTo(x + width, y, x + width, y + height, radius);
      ctx.arcTo(x + width, y + height, x, y + height, radius);
      ctx.arcTo(x, y + height, x, y, radius);
      ctx.arcTo(x, y, x + width, y, radius);
      ctx.closePath();
    }

    function update() {
      const bounds = canvas.getBoundingClientRect();
      const width = bounds.width;
      const height = bounds.height;
      if (!ctx || !width || !height) return;
      const source = video?.getAttribute("src") || video?.currentSrc || "";
      const direct = /-direct\.mp4(?:[?#]|$)/i.test(source);
      const selectedPositions = !direct && Array.isArray(sceneData?.renderPositions)
        ? sceneData.renderPositions : sceneData?.positions;
      if (selectedPositions !== activePositions) {
        activePositions = selectedPositions;
        positions = Array.isArray(selectedPositions) && selectedPositions.every((p) =>
          Array.isArray(p) && p.length >= 3 && p.slice(0, 3).every(Number.isFinite))
          ? selectedPositions.map((p) => p.slice(0, 3)) : [];
        // Dense render positions use their own chunk indices when supplied.
        const denseRender = !direct && selectedPositions === sceneData?.renderPositions;
        const chunkIndices = denseRender
          ? (sceneData?.renderChunkStarts ?? (sceneData?.chunkStarts?.length <= 1 ? sceneData.chunkStarts : []))
          : sceneData?.chunkStarts;
        starts = Array.isArray(chunkIndices)
          ? [...new Set(chunkIndices.filter((i) => Number.isInteger(i) && i >= 0 && i < positions.length))].sort((a, b) => a - b)
          : [];
        if (positions.length && starts[0] !== 0) starts.unshift(0);
      }
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const pixelWidth = Math.round(width * ratio);
      const pixelHeight = Math.round(height * ratio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const dark = isDark();
      const ink = dark ? "#dce4f3" : "#34445c";
      const subtle = dark ? "rgba(192,206,230,.23)" : "rgba(76,98,129,.19)";
      ctx.font = "500 12px system-ui, sans-serif";
      if (!positions.length) {
        ctx.fillStyle = ink;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Camera trajectory unavailable", width / 2, height / 2);
        return;
      }

      const legendColumns = Math.max(1, Math.min(starts.length, Math.floor(width / 65)));
      const legendRows = starts.length > 1 ? Math.ceil(starts.length / legendColumns) : 0;
      const legendHeight = legendRows ? legendRows * 19 + 6 : 0;
      const plotHeight = Math.max(1, height - legendHeight);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, width, plotHeight);
      ctx.clip();

      const mins = [0, 1, 2].map((axis) => Math.min(...positions.map((p) => p[axis])));
      const maxs = [0, 1, 2].map((axis) => Math.max(...positions.map((p) => p[axis])));
      const center = mins.map((value, i) => (value + maxs[i]) / 2);
      const span = Math.max(...mins.map((value, i) => maxs[i] - value), 1e-6);
      const projectRaw = (p) => {
        const x = (p[0] - center[0]) / span;
        const y = (p[1] - center[1]) / span;
        const z = (p[2] - center[2]) / span;
        return [(x - z) * 0.8, -y * 0.95 + (x + z) * 0.36];
      };
      const raw = positions.map(projectRaw);
      const xMin = Math.min(...raw.map((p) => p[0]));
      const xMax = Math.max(...raw.map((p) => p[0]));
      const yMin = Math.min(...raw.map((p) => p[1]));
      const yMax = Math.max(...raw.map((p) => p[1]));
      const scale = Math.min(Math.max(1, width - 46) / Math.max(xMax - xMin, 0.35), Math.max(1, plotHeight - 42) / Math.max(yMax - yMin, 0.35));
      const offsetX = width / 2 - (xMin + xMax) * scale / 2;
      const offsetY = plotHeight / 2 - (yMin + yMax) * scale / 2;
      const toScreen = (p) => [offsetX + p[0] * scale, offsetY + p[1] * scale];
      const point = (p) => toScreen(projectRaw(p));
      const projected = raw.map(toScreen);
      const floorY = mins[1];
      const gridExtent = span * 0.6;
      ctx.strokeStyle = subtle;
      ctx.lineWidth = 0.7;
      for (let i = 0; i <= 4; i++) {
        const value = (i / 4 * 2 - 1) * gridExtent;
        const lines = [
          [[center[0] - gridExtent, floorY, center[2] + value], [center[0] + gridExtent, floorY, center[2] + value]],
          [[center[0] + value, floorY, center[2] - gridExtent], [center[0] + value, floorY, center[2] + gridExtent]],
        ];
        for (const line of lines) {
          const a = point(line[0]);
          const b = point(line[1]);
          ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
        }
      }

      // Frame indices map each known pose to its frame in this output, rather
      // than mapping output frames to pose indices. Never infer missing timing.
      const frameIndices = direct ? sceneData?.directFrameIndices : sceneData?.renderFrameIndices;
      const fps = direct ? sceneData?.directFps : sceneData?.renderFps;
      const timing = direct
        ? (sceneData?.directTiming ?? sceneData?.directInterpolation)
        : (sceneData?.renderTiming ?? sceneData?.renderInterpolation);
      const hasTiming = Array.isArray(frameIndices) && frameIndices.length === positions.length &&
        Number.isFinite(fps) && fps > 0 && frameIndices.every((frame, i) =>
          Number.isFinite(frame) && frame >= 0 && (i === 0 || frame >= frameIndices[i - 1]));
      let completed = 0;
      let blend = 0;
      if (hasTiming) {
        const currentFrame = Math.max(0, (video?.currentTime || 0) * fps);
        while (completed < frameIndices.length - 1 && frameIndices[completed + 1] <= currentFrame) completed++;
        if ((timing === "interpolated" || timing === "linear") && completed < frameIndices.length - 1) {
          const first = frameIndices[completed];
          const last = frameIndices[completed + 1];
          blend = Math.max(0, Math.min(1, (currentFrame - first) / (last - first)));
        }
      } else {
        ctx.fillStyle = ink;
        ctx.font = "500 11px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText("Video timing unavailable", width / 2, 8);
      }
      const active = projected[completed];
      const next = projected[Math.min(completed + 1, projected.length - 1)];
      const current = [active[0] + (next[0] - active[0]) * blend, active[1] + (next[1] - active[1]) * blend];
      const chunkAt = (index) => {
        let chunk = 0;
        for (let i = 1; i < starts.length && index >= starts[i]; i++) chunk = i;
        return chunk;
      };

      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = subtle;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      projected.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.lineWidth = 3;
      for (let i = 0; i < completed; i++) {
        ctx.strokeStyle = colors[chunkAt(i) % colors.length];
        ctx.beginPath(); ctx.moveTo(...projected[i]); ctx.lineTo(...projected[i + 1]); ctx.stroke();
      }
      if (completed < projected.length - 1) {
        ctx.strokeStyle = colors[chunkAt(completed) % colors.length];
        ctx.beginPath(); ctx.moveTo(...projected[completed]); ctx.lineTo(...current); ctx.stroke();
      }
      if (starts.length > 1) {
        ctx.font = "600 10px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        starts.forEach((start, i) => {
          const p = projected[start];
          ctx.fillStyle = colors[i % colors.length];
          ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText(String(i + 1), p[0], p[1] + 0.5);
        });
      }
      ctx.fillStyle = dark ? "#fff" : "#20304b";
      ctx.strokeStyle = colors[chunkAt(completed) % colors.length];
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(current[0], current[1], 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

      // This is a video preview linked to the marker, not a recovered camera pose.
      if (video && video.readyState >= 2 && width >= 150 && plotHeight >= 105) {
        const previewWidth = Math.min(74, width * 0.25);
        const previewHeight = previewWidth * (video.videoHeight || 9) / (video.videoWidth || 16);
        const x = Math.max(5, Math.min(width - previewWidth - 5, current[0] + 12));
        const y = Math.max(5, Math.min(plotHeight - previewHeight - 5, current[1] - previewHeight - 10));
        ctx.strokeStyle = colors[chunkAt(completed) % colors.length];
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(...current); ctx.lineTo(x, y + previewHeight); ctx.stroke();
        ctx.save();
        ctx.shadowColor = "rgba(0,0,0,.25)"; ctx.shadowBlur = 8; ctx.shadowOffsetY = 3;
        roundedRect(x, y, previewWidth, previewHeight, 4);
        ctx.fillStyle = dark ? "#1b2638" : "#fff"; ctx.fill();
        ctx.shadowColor = "transparent";
        ctx.clip();
        try { ctx.drawImage(video, x, y, previewWidth, previewHeight); } catch (_) { /* Frame may not be decoded yet. */ }
        ctx.restore();
      }
      ctx.restore();

      if (legendRows) {
        const columnWidth = width / legendColumns;
        ctx.font = "600 10px system-ui, sans-serif";
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        starts.forEach((_, i) => {
          const row = Math.floor(i / legendColumns);
          const column = i % legendColumns;
          const label = "Chunk " + (i + 1);
          const itemWidth = 9 + ctx.measureText(label).width;
          const x = column * columnWidth + Math.max(3, (columnWidth - itemWidth) / 2);
          const y = plotHeight + 12 + row * 19;
          ctx.fillStyle = colors[i % colors.length];
          ctx.beginPath(); ctx.arc(x + 3, y, 3, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = ink;
          ctx.fillText(label, x + 9, y);
        });
      }
    }

    function tick() {
      frameRequest = 0;
      update();
      if (video && !video.paused && !video.ended) frameRequest = requestAnimationFrame(tick);
    }
    function start() {
      if (!frameRequest) frameRequest = requestAnimationFrame(tick);
    }
    if (video) {
      video.addEventListener("play", start);
      for (const event of ["pause", "seeked", "timeupdate", "loadedmetadata", "loadeddata", "emptied"]) {
        video.addEventListener(event, update);
      }
    }
    if ("ResizeObserver" in window) new ResizeObserver(update).observe(canvas);
    else window.addEventListener("resize", update);

    return {
      setScene(data) {
        sceneData = data;
        activePositions = null;
        positions = [];
        starts = [];
        update();
        if (video && !video.paused && !video.ended) start();
      },
      update,
    };
  };
})();
