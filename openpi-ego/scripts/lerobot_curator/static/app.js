"use strict";

const state = {
  catalog: null,
  current: null,
  filter: "all",
  videos: [],
  loaders: [],
  playing: false,
  playbackRate: 1,
  noteTimers: new Map(),
  toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  let body;
  try {
    body = await response.json();
  } catch {
    body = { error: `HTTP ${response.status}` };
  }
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function postJSON(url, payload) {
  return fetchJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("visible"), 3500);
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.0";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function shortCameraName(key) {
  return key.replace("observation.images.", "").replaceAll("_", " ").toUpperCase();
}

function filteredEpisodes() {
  if (state.filter === "all") return state.catalog.episodes;
  return state.catalog.episodes.filter((episode) => episode.status === state.filter);
}

function updateCounts() {
  const counts = { keep: 0, reject: 0, undecided: 0 };
  for (const episode of state.catalog.episodes) counts[episode.status] += 1;
  const reviewed = counts.keep + counts.reject;
  const percent = Math.round((reviewed / state.catalog.total_episodes) * 100);
  $("#keep-count").textContent = counts.keep;
  $("#reject-count").textContent = counts.reject;
  $("#undecided-count").textContent = counts.undecided;
  $("#review-progress").textContent = `${percent}%`;
  $("#progress-fill").style.width = `${percent}%`;
  $("#export-count").textContent = counts.keep;
}

function renderEpisodeList() {
  const episodes = filteredEpisodes();
  const container = $("#episode-list");
  container.replaceChildren();
  $("#filtered-count").textContent = `${episodes.length} 条`;
  if (!episodes.length) {
    const empty = document.createElement("div");
    empty.className = "list-gap";
    empty.textContent = "当前筛选条件下没有回合";
    container.append(empty);
    return;
  }

  let position = episodes.findIndex((episode) => episode.episode_index === state.current?.episode_index);
  if (position < 0) position = 0;
  const start = Math.max(0, position - 35);
  const end = Math.min(episodes.length, position + 36);
  if (start > 0) {
    const gap = document.createElement("div");
    gap.className = "list-gap";
    gap.textContent = `上方还有 ${start} 条`;
    container.append(gap);
  }
  for (const episode of episodes.slice(start, end)) {
    const button = document.createElement("button");
    button.className = "episode-row";
    button.classList.toggle("active", episode.episode_index === state.current?.episode_index);
    button.dataset.index = episode.episode_index;

    const status = document.createElement("span");
    status.className = `state ${episode.status}`;
    const copy = document.createElement("span");
    copy.className = "copy";
    const title = document.createElement("b");
    title.textContent = `Episode ${episode.episode_index}`;
    const task = document.createElement("small");
    task.textContent = episode.task;
    copy.append(title, task);
    if (episode.start_frame > 0) {
      const scissors = document.createElement("span");
      scissors.className = "trim-flag";
      scissors.textContent = `✂${episode.start_frame}`;
      scissors.title = `删除前 ${episode.start_frame} 帧`;
      copy.append(scissors);
    }
    const duration = document.createElement("span");
    duration.className = "duration";
    duration.textContent = formatTime(episode.duration).slice(0, -2);
    button.append(status, copy, duration);
    button.addEventListener("click", () => loadEpisode(episode.episode_index));
    container.append(button);
  }
  if (end < episodes.length) {
    const gap = document.createElement("div");
    gap.className = "list-gap";
    gap.textContent = `下方还有 ${episodes.length - end} 条`;
    container.append(gap);
  }
  requestAnimationFrame(() => container.querySelector(".episode-row.active")?.scrollIntoView({ block: "nearest" }));
}

function setupVideoGrid() {
  const grid = $("#video-grid");
  grid.replaceChildren();
  const videoCount = state.catalog.video_keys.length;
  const columns = Math.min(videoCount, 2);
  grid.style.setProperty("--video-columns", columns);
  grid.style.setProperty("--video-rows", Math.ceil(videoCount / columns));
  state.videos = [];
  state.loaders = [];
  for (const [index, key] of state.catalog.video_keys.entries()) {
    const cell = document.createElement("div");
    cell.className = "video-cell";
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.disablePictureInPicture = true;
    video.dataset.key = key;
    video.addEventListener("loadeddata", () => state.loaders[index].classList.add("hidden"));
    video.addEventListener("error", () => showToast(`视频读取失败：${shortCameraName(key)}`, true), { passive: true });
    video.addEventListener("dblclick", () => cell.requestFullscreen?.());
    const label = document.createElement("span");
    label.className = "camera-label";
    label.textContent = shortCameraName(key);
    const loader = document.createElement("span");
    loader.className = "video-loading";
    loader.textContent = "载入视频…";
    cell.append(video, label, loader);
    grid.append(cell);
    state.videos.push(video);
    state.loaders.push(loader);
  }

  const master = state.videos[0];
  master.addEventListener("timeupdate", updateTimeline);
  master.addEventListener("durationchange", updateTimeline);
  master.addEventListener("play", () => setPlayingUI(true));
  master.addEventListener("pause", () => setPlayingUI(false));
  master.addEventListener("ended", () => setPlayingUI(false));
}

function setPlayingUI(playing) {
  state.playing = playing;
  $("#play-icon").textContent = playing ? "❚❚" : "▶";
}

async function playAll() {
  const readyVideos = state.videos.filter((video) => video.readyState > 0);
  await Promise.allSettled(
    readyVideos.map((video) => {
      video.playbackRate = state.playbackRate;
      return video.play();
    }),
  );
  setPlayingUI(!state.videos[0].paused);
}

function pauseAll() {
  for (const video of state.videos) video.pause();
  setPlayingUI(false);
}

function togglePlayback() {
  if (state.videos[0].paused) playAll();
  else pauseAll();
}

function updateTimeline() {
  const master = state.videos[0];
  if (!master || !state.current) return;
  const duration = Number.isFinite(master.duration) ? master.duration : state.current.duration;
  const progress = duration > 0 ? master.currentTime / duration : 0;
  $("#timeline").value = Math.round(progress * 1000);
  $("#current-time").textContent = formatTime(master.currentTime);
  $("#total-time").textContent = formatTime(duration);
  updateTrimReadout();
}

function currentFrame() {
  const master = state.videos[0];
  const fps = state.catalog.fps;
  const frame = Math.round((master?.currentTime || 0) * fps);
  const last = (state.current?.length || 1) - 1;
  return Math.max(0, Math.min(last, frame));
}

function updateTrimReadout() {
  if (!state.current) return;
  const fps = state.catalog.fps;
  const start = state.current.start_frame || 0;
  const startText = start > 0 ? `删除前 ${start} 帧 (t=${(start / fps).toFixed(2)}s)` : "未设置起点";
  $("#trim-readout").textContent = `当前帧 ${currentFrame()} · ${startText}`;
}

async function setTrimStart() {
  if (!state.current) return;
  const episode = state.current;
  const startFrame = currentFrame();
  try {
    await postJSON("/api/trim", { episode_index: episode.episode_index, start_frame: startFrame });
    episode.start_frame = startFrame;
    updateEpisodeHeader();
    updateTrimReadout();
    renderEpisodeList();
    showToast(`Episode ${episode.episode_index}：起点设为第 ${startFrame} 帧`);
  } catch (error) {
    showToast(`设置起点失败：${error.message}`, true);
  }
}

async function setTrimStartSeconds() {
  if (!state.current) return;
  const seconds = Number($("#trim-seconds").value);
  if (!Number.isFinite(seconds) || seconds < 0) {
    showToast("请输入有效的秒数", true);
    return;
  }
  const episode = state.current;
  const last = episode.length - 1;
  const startFrame = Math.max(0, Math.min(last, Math.round(seconds * state.catalog.fps)));
  try {
    await postJSON("/api/trim", { episode_index: episode.episode_index, start_frame: startFrame });
    episode.start_frame = startFrame;
    updateEpisodeHeader();
    updateTrimReadout();
    renderEpisodeList();
    showToast(`Episode ${episode.episode_index}：起点设为 ${seconds}s（第 ${startFrame} 帧）`);
  } catch (error) {
    showToast(`设置起点失败：${error.message}`, true);
  }
}

async function clearTrimStart() {
  if (!state.current || !(state.current.start_frame > 0)) return;
  const episode = state.current;
  try {
    await postJSON("/api/trim", { episode_index: episode.episode_index, start_frame: 0 });
    episode.start_frame = 0;
    updateEpisodeHeader();
    updateTrimReadout();
    renderEpisodeList();
    showToast(`Episode ${episode.episode_index}：已清除起点`);
  } catch (error) {
    showToast(`清除起点失败：${error.message}`, true);
  }
}

function gotoTrimStart() {
  if (!state.current) return;
  const duration = state.current.duration || 1;
  seekToFraction((state.current.start_frame || 0) / state.catalog.fps / duration);
}

function stepFrame(delta) {
  const master = state.videos[0];
  if (!master || !state.current) return;
  pauseAll();
  const duration = Number.isFinite(master.duration) ? master.duration : state.current.duration;
  const time = Math.max(0, Math.min(duration, master.currentTime + delta / state.catalog.fps));
  for (const video of state.videos) {
    if (video.readyState > 0) video.currentTime = Math.min(time, video.duration || time);
  }
  updateTimeline();
}

function seekToFraction(fraction) {
  const master = state.videos[0];
  const duration = Number.isFinite(master.duration) ? master.duration : state.current.duration;
  const time = Math.max(0, Math.min(duration, duration * fraction));
  for (const video of state.videos) {
    if (video.readyState > 0) video.currentTime = Math.min(time, video.duration || time);
  }
  updateTimeline();
}

function synchronizeVideos() {
  const master = state.videos[0];
  if (master && !master.paused && !master.ended) {
    for (const follower of state.videos.slice(1)) {
      follower.playbackRate = state.playbackRate;
      if (follower.readyState >= 2 && Math.abs(follower.currentTime - master.currentTime) > 0.12) {
        follower.currentTime = master.currentTime;
      }
      if (follower.paused && follower.readyState >= 2) follower.play().catch(() => {});
    }
  }
  requestAnimationFrame(synchronizeVideos);
}

function updateEpisodeHeader() {
  const episode = state.current;
  $("#episode-index").textContent = episode.episode_index;
  $("#task-name").textContent = episode.task;
  $("#frame-count").textContent = episode.length.toLocaleString();
  $("#duration-value").textContent = `${episode.duration.toFixed(1)} s`;
  $("#fps-value").textContent = `${state.catalog.fps} FPS`;
  const start = episode.start_frame || 0;
  const trimValue = $("#trim-value");
  trimValue.textContent = start > 0 ? `删除前 ${start} 帧` : "未设置";
  trimValue.classList.toggle("is-set", start > 0);
  $("#episode-note").value = episode.note || "";

  const pill = $("#status-pill");
  const labels = { undecided: "未处理", keep: "已保留", reject: "已排除" };
  pill.textContent = labels[episode.status];
  pill.className = `status-pill ${episode.status}`;
  $("#mark-keep").classList.toggle("active", episode.status === "keep");
  $("#mark-reject").classList.toggle("active", episode.status === "reject");
}

function loadEpisode(episodeIndex, autoplay = state.playing) {
  const episode = state.catalog.episodes.find((item) => item.episode_index === Number(episodeIndex));
  if (!episode) {
    showToast(`找不到 Episode ${episodeIndex}`, true);
    return;
  }
  pauseAll();
  state.current = episode;
  updateEpisodeHeader();
  renderEpisodeList();
  $("#timeline").value = 0;
  $("#current-time").textContent = "00:00.0";
  $("#total-time").textContent = formatTime(episode.duration);
  updateTrimReadout();

  state.videos.forEach((video, index) => {
    state.loaders[index].classList.remove("hidden");
    const key = state.catalog.video_keys[index];
    video.src = `/api/video?episode=${episode.episode_index}&key=${encodeURIComponent(key)}`;
    video.playbackRate = state.playbackRate;
    video.load();
  });
  if (autoplay) {
    state.videos[0].addEventListener("canplay", () => playAll(), { once: true });
  }
}

function adjacentEpisode(direction, onlyUndecided = false) {
  const source = onlyUndecided
    ? state.catalog.episodes.filter((episode) => episode.status === "undecided")
    : filteredEpisodes();
  if (!source.length) return null;
  const currentIndex = source.findIndex((episode) => episode.episode_index === state.current.episode_index);
  if (currentIndex < 0) return direction > 0 ? source[0] : source.at(-1);
  const nextIndex = currentIndex + direction;
  if (nextIndex < 0 || nextIndex >= source.length) return null;
  return source[nextIndex];
}

function navigate(direction) {
  const episode = adjacentEpisode(direction);
  if (episode) loadEpisode(episode.episode_index);
  else showToast(direction > 0 ? "已经是最后一个回合" : "已经是第一个回合");
}

async function saveDecision(status) {
  if (!state.current) return;
  const episode = state.current;
  const note = $("#episode-note").value;
  const nextUndecided = status !== "undecided" && $("#auto-next").checked ? adjacentEpisode(1, true) : null;
  clearTimeout(state.noteTimers.get(episode.episode_index));
  try {
    await postJSON("/api/decision", { episode_index: episode.episode_index, status, note });
    episode.status = status;
    episode.note = note.trim();
    updateCounts();
    updateEpisodeHeader();
    renderEpisodeList();
    if (nextUndecided && nextUndecided.episode_index !== episode.episode_index) loadEpisode(nextUndecided.episode_index);
  } catch (error) {
    showToast(`保存失败：${error.message}`, true);
  }
}

function queueNoteSave() {
  const episode = state.current;
  if (!episode) return;
  clearTimeout(state.noteTimers.get(episode.episode_index));
  const note = $("#episode-note").value;
  const timer = setTimeout(async () => {
    try {
      await postJSON("/api/decision", {
        episode_index: episode.episode_index,
        status: episode.status,
        note,
      });
      episode.note = note.trim();
    } catch (error) {
      showToast(`备注保存失败：${error.message}`, true);
    } finally {
      state.noteTimers.delete(episode.episode_index);
    }
  }, 550);
  state.noteTimers.set(episode.episode_index, timer);
}

function setFilter(filter) {
  state.filter = filter;
  $$("#filter-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.filter === filter));
  const episodes = filteredEpisodes();
  if (episodes.length && !episodes.some((episode) => episode.episode_index === state.current.episode_index)) {
    loadEpisode(episodes[0].episode_index, false);
  } else {
    renderEpisodeList();
  }
}

function jumpToEpisode() {
  const value = Number($("#jump-input").value);
  if (!Number.isInteger(value)) return;
  setFilter("all");
  loadEpisode(value, false);
}

function openExportDialog() {
  updateCounts();
  const trimmed = state.catalog.episodes.filter((episode) => episode.status === "keep" && episode.start_frame > 0).length;
  $("#export-trim-count").textContent = trimmed;
  $("#output-path").value ||= `${state.catalog.default_output}_trimmed`;
  $("#export-warning").textContent = "";
  $("#export-dialog").showModal();
}

async function runExport() {
  const button = $("#run-export");
  const output = $("#output-path").value.trim();
  const codec = $("input[name='export-codec']:checked").value;
  const jobs = $("#export-jobs").value.trim();
  const crf = $("#export-crf").value.trim();
  if (!output) {
    $("#export-warning").textContent = "请填写输出目录。";
    return;
  }
  button.disabled = true;
  button.textContent = "正在裁剪并重编码…";
  $("#export-warning").textContent = "正在重编码视频，耗时取决于回合数与核数；请不要关闭此页面或服务。";
  try {
    const payload = { output, video_codec: codec };
    if (jobs) payload.jobs = Number(jobs);
    if (crf) payload.crf = Number(crf);
    const result = await postJSON("/api/export", payload);
    $("#export-dialog").close();
    showToast(
      `导出完成：${result.episodes} 回合（${result.trimmed_episodes} 条已裁剪），${result.frames.toLocaleString()} 帧 → ${result.output}`,
    );
  } catch (error) {
    $("#export-warning").textContent = `导出失败：${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "开始导出";
  }
}

function bindEvents() {
  $("#play-toggle").addEventListener("click", togglePlayback);
  $("#previous-episode").addEventListener("click", () => navigate(-1));
  $("#next-episode").addEventListener("click", () => navigate(1));
  $("#timeline").addEventListener("input", (event) => seekToFraction(Number(event.target.value) / 1000));
  $("#speed-select").addEventListener("change", (event) => {
    state.playbackRate = Number(event.target.value);
    for (const video of state.videos) video.playbackRate = state.playbackRate;
  });
  $("#mark-keep").addEventListener("click", () => saveDecision("keep"));
  $("#mark-reject").addEventListener("click", () => saveDecision("reject"));
  $("#mark-undecided").addEventListener("click", () => saveDecision("undecided"));
  $("#set-trim").addEventListener("click", setTrimStart);
  $("#clear-trim").addEventListener("click", clearTrimStart);
  $("#goto-trim").addEventListener("click", gotoTrimStart);
  $("#set-trim-seconds").addEventListener("click", setTrimStartSeconds);
  $("#trim-seconds").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      setTrimStartSeconds();
    }
  });
  $("#trim-back").addEventListener("click", () => stepFrame(-1));
  $("#trim-forward").addEventListener("click", () => stepFrame(1));
  $("#episode-note").addEventListener("input", queueNoteSave);
  $("#jump-button").addEventListener("click", jumpToEpisode);
  $("#jump-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") jumpToEpisode();
  });
  $$("#filter-tabs button").forEach((button) => button.addEventListener("click", () => setFilter(button.dataset.filter)));
  $("#open-export").addEventListener("click", openExportDialog);
  $("#run-export").addEventListener("click", runExport);

  window.addEventListener("keydown", (event) => {
    const tag = event.target.tagName;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || $("#export-dialog").open) return;
    if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowLeft") navigate(-1);
    else if (event.key === "ArrowRight") navigate(1);
    else if (event.key.toLowerCase() === "k") saveDecision("keep");
    else if (event.key.toLowerCase() === "x") saveDecision("reject");
    else if (event.key.toLowerCase() === "u") saveDecision("undecided");
    else if (event.key.toLowerCase() === "s") setTrimStart();
    else if (event.key.toLowerCase() === "d") clearTrimStart();
    else if (event.key.toLowerCase() === "g") gotoTrimStart();
    else if (event.key === ",") {
      event.preventDefault();
      stepFrame(-1);
    } else if (event.key === ".") {
      event.preventDefault();
      stepFrame(1);
    }
  });
}

async function initialize() {
  try {
    state.catalog = await fetchJSON("/api/catalog");
    $("#dataset-name").textContent = state.catalog.name;
    $("#dataset-path").textContent = state.catalog.root;
    $("#fps-value").textContent = `${state.catalog.fps} FPS`;
    $("#jump-input").max = Math.max(...state.catalog.episodes.map((episode) => episode.episode_index));
    $("#output-path").value = state.catalog.default_output;
    setupVideoGrid();
    bindEvents();
    updateCounts();
    if (!state.catalog.episodes.length) throw new Error("数据集没有 episodes");
    loadEpisode(state.catalog.episodes[0].episode_index, false);
    requestAnimationFrame(synchronizeVideos);
  } catch (error) {
    showToast(`初始化失败：${error.message}`, true);
    $("#dataset-name").textContent = "数据集加载失败";
  }
}

initialize();
