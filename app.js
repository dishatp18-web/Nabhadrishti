/* =====================================================================
   BAS Payload Operations Copilot - HUD client
   Connects to /ws/telemetry and mutates ONLY the DOM panels + SVG
   overlay. The video element (img src=/video_feed) is never touched,
   so telemetry can never clobber or reload the stream.
   ===================================================================== */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    boot: $("boot-overlay"),
    bootStatus: $("boot-status"),
    sysState: $("sys-state"),
    sysStateText: $("sys-state-text"),
    utcClock: $("utc-clock"),

    curIdx: $("current-step-idx"),
    curName: $("current-step-name"),
    curDesc: $("current-step-desc"),
    curProg: $("current-step-progress"),
    aiLine: $("ai-line"),
    aiLineText: $("ai-line-text"),
    aiHoldBadge: $("ai-hold-badge"),
    recDot: $("rec-dot"),
    stepList: $("step-list"),

    healthCamera: $("health-camera"),
    healthSource: $("health-source"),
    healthDetector: $("health-detector"),
    healthHands: $("health-hands"),
    healthVoice: $("health-voice"),
    healthAction: $("health-action"),
    healthRecording: $("health-recording"),
    healthFps: $("health-fps"),
    healthRes: $("health-res"),
    healthUptime: $("health-uptime"),

    viewport: $("viewport"),
    video: $("video-feed"),
    svg: $("overlay-svg"),
    toggleBoxes: $("toggle-boxes"),
    camSelect: $("cam-select"),
    wsStatus: $("ws-status"),
    alertFlash: $("alert-flash"),
    videoError: $("video-error"),

    frameCounter: $("frame-counter"),
    handCounter: $("hand-counter"),
    objectCounter: $("object-counter"),
    interactionCounter: $("interaction-counter"),

    trackList: $("track-list"),
    alertFeed: $("alert-feed"),
    alertCounterBadge: $("alert-counter-badge"),

    btnOpenExperiments: $("btn-open-experiments"),
    btnToolbarExperiments: $("btn-toolbar-experiments"),
    btnCloseExperiments: $("btn-close-experiments"),
    btnRefreshExperiments: $("btn-refresh-experiments"),
    expModal: $("experiments-modal"),
    expCountBadge: $("exp-count-badge"),
    expList: $("exp-list"),
    expDetailPane: $("exp-detail-pane"),
    expSearchInput: $("exp-search-input"),
  };

  const SVG_NS = "http://www.w3.org/2000/svg";

  let lastAlertHash = "";
  let renderedAlertIds = [];
  let alertIdsSeen = new Set();
  let lastSeq = 0;
  let experimentsData = [];
  let selectedSessionId = null;
  let activeFilter = "all";

  /* ------------------------------------------------ clock */
  setInterval(() => {
    const now = new Date();
    els.utcClock.textContent =
      now.toISOString().slice(11, 19) + " UTC";
  }, 500);

  /* ------------------------------------------------ video resilience
     If the MJPEG stream dies (server restart etc.) the <img> fires an
     error event; we re-attach the same URL with a cache-buster until it
     comes back. The element itself is never reloaded while healthy. */
  function wireVideoRecovery() {
    let retries = 0;
    els.video.addEventListener("error", () => {
      retries += 1;
      els.videoError.classList.remove("hidden");
      const backoff = Math.min(500 * retries, 4000);
      setTimeout(() => {
        els.video.src = "/video_feed?r=" + Date.now();
      }, backoff);
    });
    els.video.addEventListener("load", () => {
      retries = 0;
      els.videoError.classList.add("hidden");
    });
  }

  /* ------------------------------------------------ SVG overlay helpers
     Telemetry boxes are normalized (0..1) over the full FRAME. The <img>
     uses object-fit: contain, so the frame may be letterboxed inside the
     viewport. We compute the actual displayed-content rectangle and map
     normalized frame coords onto it, so boxes hug the video perfectly. */
  function contentRect() {
    const vp = els.video.getBoundingClientRect();
    const vw = els.video.naturalWidth || 16;
    const vh = els.video.naturalHeight || 9;
    const scale = Math.min(vp.width / vw, vp.height / vh);
    const w = vw * scale, h = vh * scale;
    return {
      x: (vp.width - w) / 2,
      y: (vp.height - h) / 2,
      w, h,
    };
  }

  function clearSvg() {
    while (els.svg.firstChild) els.svg.removeChild(els.svg.firstChild);
  }

  function svgTag(name, attrs, textContent) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (textContent) node.textContent = textContent;
    return node;
  }

  function drawOverlay(t) {
    clearSvg();
    if (!els.toggleBoxes.checked) return;
    const rect = contentRect();
    if (rect.w <= 0) return;

    // Use the element's own pixel space as the SVG user space, so mapped
    // coordinates land exactly on the letterboxed video content area.
    const box = els.svg.getBoundingClientRect();
    if (box.width <= 0 || box.height <= 0) return;
    els.svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    els.svg.setAttribute("preserveAspectRatio", "none");

    const map = (nx, ny) => [rect.x + nx * rect.w, rect.y + ny * rect.h];

    for (const obj of t.objects ?? []) {
      const b = obj.box;
      const [x0, y0] = map(b.x, b.y);
      const [x1, y1] = map(b.x + b.w, b.y + b.h);
      const held = obj.held_by_hand_id != null;
      const stroke = held ? "#ff4655" : "#ffb545";
      // No SVG text: labels are baked into the MJPEG frames as readable
      // chips, so the HUD only draws geometry (avoids double text).
      els.svg.appendChild(svgTag("rect", {
        x: x0, y: y0, width: x1 - x0, height: y1 - y0,
        fill: "none", stroke,
        "stroke-width": 2,
        "vector-effect": "non-scaling-stroke",
        rx: 6,
      }));
    }

    for (const hand of t.hands ?? []) {
      const b = hand.box;
      const [x0, y0] = map(b.x, b.y);
      const [x1, y1] = map(b.x + b.w, b.y + b.h);
      els.svg.appendChild(svgTag("rect", {
        x: x0, y: y0, width: x1 - x0, height: y1 - y0,
        fill: "none", stroke: "#43f59b",
        "stroke-width": 1.5, "stroke-dasharray": "10 6",
        "vector-effect": "non-scaling-stroke", rx: 8,
      }));
      if (hand.landmarks) {
        for (const lm of hand.landmarks) {
          const [px, py] = map(lm.x, lm.y);
          els.svg.appendChild(svgTag("circle", {
            cx: px, cy: py, r: 2.5, fill: "#a68bff", opacity: 0.75,
          }));
        }
      }
    }
  }

  /* ------------------------------------------------ panels */
  function renderProtocol(t) {
    const steps = t.protocol ?? [];
    const cur = t.current_step_index ?? 0;

    // Step list
    const frag = document.createDocumentFragment();
    const glyphs = ["◇", "◈", "◆"];
    steps.forEach((s, i) => {
      const li = document.createElement("li");
      const glyph = document.createElement("span");
      glyph.className = "glyph";
      const name = document.createElement("span");
      name.className = "s-name";
      name.textContent = `${i + 1}. ${s.name}`;
      li.appendChild(glyph);
      li.appendChild(name);

      if (s.status === "complete") { li.classList.add("complete"); glyph.textContent = "✔"; }
      else if (s.status === "active") { li.classList.add("active"); glyph.textContent = "◈"; }
      else { glyph.textContent = glyphs[Math.min(i, 2)] ?? "◇"; }
      frag.appendChild(li);
    });
    els.stepList.replaceChildren(frag);

    // Current step card
    const active = steps.find((s) => s.index === cur);
    const finished = steps.length > 0 && steps.every((s) => s.status === "complete");
    if (finished) {
      els.curIdx.textContent = "PROTOCOL COMPLETE";
      els.curName.textContent = "All steps verified";
      els.curDesc.textContent = "Payload operations sequence finished. Awaiting reset.";
      els.curProg.style.width = "100%";
    } else if (active) {
      els.curIdx.textContent = `STEP ${cur + 1}/${steps.length}`;
      els.curName.textContent = active.name;
      els.curDesc.textContent = active.description || "";
      els.curProg.style.width = `${Math.round((active.progress ?? 0) * 100)}%`;
    }
  }

  function renderHealth(t) {
    const h = t.health ?? {};
    els.healthCamera.textContent = (h.camera ?? "—").toUpperCase();
    els.healthCamera.parentElement.className =
      /LIVE|SIMULATED/.test(h.camera ?? "") ? "ok" : "warn";
    els.healthSource.textContent = h.camera_source || "—";
    els.healthDetector.textContent = (h.detector ?? "—").toUpperCase();
    els.healthHands.textContent = (h.hands_model ?? "—").toUpperCase();
    els.healthVoice.textContent = (h.voice ?? "—").toUpperCase();
    els.healthAction.textContent = (h.action_model ?? "—").toUpperCase();
    if (h.recording_file) {
      els.healthRecording.textContent = `● ${h.recording_file}`;
      els.healthRecording.parentElement.className = "warn";
      els.recDot.classList.remove("hidden");
    } else {
      els.healthRecording.textContent = "—";
      els.recDot.classList.add("hidden");
    }
    els.healthFps.textContent = (h.fps ?? 0).toFixed(1);
    els.healthRes.textContent = h.frame_width ? `${h.frame_width}×${h.frame_height}` : "—";
    const up = h.uptime_s ?? 0;
    const mm = String(Math.floor(up / 60)).padStart(2, "0");
    const ss = String(Math.floor(up % 60)).padStart(2, "0");
    els.healthUptime.textContent = `${mm}:${ss}`;
  }

  function renderAi(t) {
    if (!t.ai_action) {
      els.aiLine.dataset.state = "idle";
      els.aiLineText.textContent = "AI: standby (FSM-only mode)";
      els.aiHoldBadge.classList.add("hidden");
      return;
    }
    const pct = Math.round((t.ai_confidence ?? 0) * 100);
    els.aiLineText.textContent =
      `AI: ${t.ai_action.toUpperCase()} (${pct}%) · ` +
      `${t.ai_agrees === false ? "MISMATCH" : "AGREES"}`;
    els.aiLine.dataset.state = t.ai_agrees === false ? "disagree" : "agree";
    els.aiHoldBadge.classList.toggle("hidden", !t.hold);
  }

  function renderState(t) {
    const state = t.state ?? "nominal";
    els.sysState.dataset.state = state;
    els.sysStateText.textContent = state.toUpperCase();
    if (els.viewport) {
      els.viewport.dataset.state = state;
    }
  }

  function triggerAlertPulse(level) {
    if (!els.alertFlash) return;
    const cls = level === "critical" ? "flash-crit" : "flash-warn";
    els.alertFlash.className = "alert-flash";
    // Force reflow to re-trigger CSS keyframe cleanly
    void els.alertFlash.offsetWidth;
    els.alertFlash.className = `alert-flash ${cls}`;
    setTimeout(() => {
      if (els.alertFlash && els.alertFlash.classList.contains(cls)) {
        els.alertFlash.className = "alert-flash";
      }
    }, level === "critical" ? 1300 : 1100);
  }

  function renderTracks(t) {
    const objs = t.objects ?? [];
    const hands = t.hands ?? [];
    const inters = t.interactions ?? [];

    els.handCounter.textContent = `HANDS ${hands.length}`;
    els.objectCounter.textContent = `OBJ ${objs.length}`;
    els.interactionCounter.textContent = `INT ${inters.length}`;
    els.frameCounter.textContent = `FRM ${t.seq ?? "—"}`;

    if (objs.length === 0) {
      els.trackList.replaceChildren(
        el("div", "empty-hint mono", "NO CONTACTS"));
      return;
    }
    const frag = document.createDocumentFragment();
    for (const o of objs) {
      const card = el("div", "track-card");
      const kinds = inters.filter((i) => i.object_id === o.object_id).map((i) => i.kind);
      if (o.held_by_hand_id != null) card.classList.add("held");

      const head = el("div", "t-head");
      head.appendChild(el("span", "", prettyLabel(o.label)));
      head.appendChild(el("span", "t-id", `#${o.object_id}`));
      card.appendChild(head);

      const meta = el("div", "t-meta");
      meta.appendChild(el("span", "",
        `CONF ${(o.confidence * 100).toFixed(0)}%`));
      meta.appendChild(el("span", "",
        o.nearest_hand_dist != null
          ? `HAND Δ ${(o.nearest_hand_dist * 100).toFixed(0)}`
          : "HAND Δ —"));
      card.appendChild(meta);

      if (o.held_by_hand_id != null || kinds.length > 0) {
        const badges = el("div", "t-meta");
        if (kinds.includes("lift") || o.held_by_hand_id != null)
          badges.appendChild(el("span", "badge lift", "LIFT/HELD"));
        else if (kinds.includes("contact"))
          badges.appendChild(el("span", "badge contact", "CONTACT"));
        else if (kinds.includes("near"))
          badges.appendChild(el("span", "badge near", "PROXIMITY"));
        card.appendChild(badges);
      }
      frag.appendChild(card);
    }
    els.trackList.replaceChildren(frag);
  }

  function renderAlerts(t) {
    const alerts = (t.alerts ?? []).slice(0, 30);

    // Compute signature to avoid ANY DOM thrashing/jittering if alerts didn't change
    const signature = alerts.map((a) => `${a.id}:${a.level}:${a.code}`).join("|");
    if (signature === lastAlertHash) {
      return; // Nothing changed, keep DOM 100% still without redrawing!
    }
    lastAlertHash = signature;

    // Check for genuinely fresh alerts that were never seen before
    const fresh = alerts.filter((a) => !alertIdsSeen.has(a.id));
    if (fresh.length > 0) {
      for (const a of fresh) alertIdsSeen.add(a.id);
      const worst = fresh.some((a) => a.level === "critical")
        ? "critical"
        : fresh.some((a) => a.level === "warning")
        ? "warning"
        : null;
      if (worst) {
        triggerAlertPulse(worst);
      }
    }
    if (alertIdsSeen.size > 200) alertIdsSeen = new Set([...alertIdsSeen].slice(-100));

    // Update panel badge
    if (els.alertCounterBadge) {
      if (alerts.length === 0) {
        els.alertCounterBadge.textContent = "0 NOMINAL";
        els.alertCounterBadge.className = "panel-badge mono";
      } else {
        const crit = alerts.some((a) => a.level === "critical");
        const warn = alerts.some((a) => a.level === "warning");
        els.alertCounterBadge.textContent = `${alerts.length} ${crit ? "CRITICAL" : warn ? "WARNING" : "ACTIVE"}`;
        els.alertCounterBadge.className = `panel-badge mono ${crit ? "crit" : warn ? "warn" : ""}`;
      }
    }

    if (alerts.length === 0) {
      renderedAlertIds = [];
      els.alertFeed.replaceChildren(el("div", "empty-hint mono", "ALL SYSTEMS NOMINAL"));
      return;
    }

    // Build fragment: only brand-new alerts receive the .alert-new entry animation
    const frag = document.createDocumentFragment();
    for (const a of alerts) {
      const isNew = fresh.some((f) => f.id === a.id);
      const item = el("div", `alert-item ${a.level}${isNew ? " alert-new" : ""}`);
      item.dataset.alertId = a.id;
      const head = el("div", "a-head");
      head.appendChild(el("span", "a-code", a.code));
      head.appendChild(
        el("span", "a-time", new Date(a.ts * 1000).toISOString().slice(11, 19) + "Z")
      );
      item.appendChild(head);
      item.appendChild(el("div", "a-msg", a.message));
      frag.appendChild(item);
    }
    renderedAlertIds = alerts.map((a) => a.id);
    els.alertFeed.replaceChildren(frag);
  }

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function prettyLabel(label) {
    return (label.startsWith("gesture_")
      ? label.slice("gesture_".length) : label).toUpperCase();
  }

  /* ------------------------------------------------ telemetry socket */
  let ws = null;
  let wsRetry = 0;

  function connectTelemetry() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);

    ws.onopen = () => {
      wsRetry = 0;
      els.wsStatus.textContent = "WS:LINK";
      els.wsStatus.className = "link-chip ok";
      els.bootStatus.textContent = "Uplink established.";
    };

    ws.onmessage = (evt) => {
      let t;
      try { t = JSON.parse(evt.data); } catch { return; }
      lastSeq = t.seq ?? lastSeq;
      renderState(t);
      renderProtocol(t);
      renderHealth(t);
      renderAi(t);
      renderTracks(t);
      renderAlerts(t);
      drawOverlay(t);
    };

    ws.onclose = () => {
      wsRetry += 1;
      els.wsStatus.textContent = "WS:LOST";
      els.wsStatus.className = "link-chip err";
      const backoff = Math.min(500 * wsRetry, 5000);
      setTimeout(connectTelemetry, backoff);
    };

    ws.onerror = () => { try { ws.close(); } catch { /* noop */ } };
  }

  /* ------------------------------------------------ camera switcher
     Populates the dropdown from /api/cameras and hot-swaps the source via
     POST /api/camera/select. The MJPEG <img> self-heals via its error
     handler: the switch briefly interrupts the stream and the recovery
     logic re-attaches the feed once frames flow again. */
  async function loadCameras() {
    try {
      const res = await fetch("/api/cameras");
      const data = await res.json();
      const frag = document.createDocumentFragment();
      for (const cam of data.cameras ?? []) {
        const opt = document.createElement("option");
        opt.value = cam.id;
        opt.textContent = "CAM: " + cam.name.toUpperCase();
        frag.appendChild(opt);
      }
      const sim = document.createElement("option");
      sim.value = "sim";
      sim.textContent = "CAM: SYNTHETIC TEST PATTERN";
      frag.appendChild(sim);
      els.camSelect.replaceChildren(frag);
      els.camSelect.value = data.current ?? "";
      if (els.camSelect.selectedIndex < 0) els.camSelect.selectedIndex = 0;
    } catch { /* dropdown stays with placeholder; retried on next select */ }
  }

  async function onCameraSelect() {
    const wanted = els.camSelect.value;
    els.camSelect.disabled = true;
    try {
      const res = await fetch("/api/camera/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: wanted }),
      });
      const data = await res.json();
      if (!data.ok) {
        // Backend restored the previous source; reflect reality.
        els.wsStatus.textContent = "CAM:SW FAIL";
        els.wsStatus.className = "link-chip err";
        setTimeout(() => { els.wsStatus.textContent = "WS:LINK";
                           els.wsStatus.className = "link-chip ok"; }, 2500);
      }
    } catch { /* network hiccup - dropdown refreshed below anyway */ }
    els.camSelect.disabled = false;
    await loadCameras();
    // Nudge the video element to re-attach the (new) stream promptly.
    els.video.src = "/video_feed?r=" + Date.now();
  }

  /* ------------------------------------------------ Previous Experiments Archive System */
  async function loadExperiments(autoSelect = true) {
    try {
      if (els.expList && els.expList.children.length === 0) {
        els.expList.innerHTML = `<div class="loading-state mono">Fetching mission archive…</div>`;
      }
      const res = await fetch("/api/experiments");
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      experimentsData = data.experiments || [];

      if (els.expCountBadge) {
        els.expCountBadge.textContent = experimentsData.length;
      }

      renderExperimentList();

      if (autoSelect && experimentsData.length > 0) {
        const toSelect = selectedSessionId
          ? experimentsData.find((e) => e.session_id === selectedSessionId)?.session_id
          : experimentsData[0].session_id;
        selectExperiment(toSelect || experimentsData[0].session_id);
      }
    } catch (err) {
      console.warn("Could not load experiments:", err);
      if (els.expList) {
        els.expList.innerHTML = `<div class="empty-hint mono">ERROR LOADING MISSION ARCHIVE</div>`;
      }
    }
  }

  function renderExperimentList() {
    if (!els.expList) return;
    const search = (els.expSearchInput?.value || "").toLowerCase().trim();

    const filtered = experimentsData.filter((exp) => {
      // Filter tab
      if (activeFilter === "completed" && exp.status !== "COMPLETED" && exp.status !== "RESOLVED") return false;
      if (activeFilter === "active" && !exp.is_current) return false;
      if (activeFilter === "anomalies" && exp.anomalies_count === 0 && exp.status !== "ANOMALY / HOLD") return false;

      // Search query
      if (search) {
        const str = `${exp.session_id} ${exp.protocol} ${exp.status}`.toLowerCase();
        if (!str.includes(search)) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      els.expList.innerHTML = `<div class="empty-hint mono">NO EXPERIMENTS MATCH FILTER</div>`;
      return;
    }

    const frag = document.createDocumentFragment();
    for (const exp of filtered) {
      const card = el("div", `exp-card${exp.session_id === selectedSessionId ? " selected" : ""}`);
      card.dataset.sessionId = exp.session_id;

      const header = el("div", "exp-card-header");
      const title = el("span", "exp-card-title", exp.session_id);
      const statusClass = exp.is_current ? "active" : (exp.status === "COMPLETED" ? "completed" : exp.status === "RESOLVED" ? "resolved" : "hold");
      const statusPill = el("span", `exp-status-pill ${statusClass}`, exp.is_current ? "● LIVE" : exp.status);
      header.appendChild(title);
      header.appendChild(statusPill);
      card.appendChild(header);

      const meta = el("div", "exp-card-meta mono");
      const protoName = exp.protocol.toUpperCase();
      meta.appendChild(el("span", "", `${protoName} PROTOCOL`));
      meta.appendChild(el("span", "", exp.duration_str || "0s"));
      card.appendChild(meta);

      const badges = el("div", "exp-card-badges mono");
      badges.appendChild(el("span", "exp-mini-badge highlight", `STEPS: ${exp.steps_completed}/${exp.total_protocol_steps}`));
      if (exp.anomalies_count > 0) {
        badges.appendChild(el("span", "exp-mini-badge", `⚠ ${exp.anomalies_count} ALERTS`));
      } else {
        badges.appendChild(el("span", "exp-mini-badge", `✔ NOMINAL`));
      }
      if (exp.has_video) {
        badges.appendChild(el("span", "exp-mini-badge", `📹 ${exp.video_size_mb} MB`));
      }
      card.appendChild(badges);

      card.addEventListener("click", () => {
        selectExperiment(exp.session_id);
      });

      frag.appendChild(card);
    }
    els.expList.replaceChildren(frag);
  }

  async function selectExperiment(sessionId) {
    selectedSessionId = sessionId;
    document.querySelectorAll(".exp-card").forEach((c) => {
      c.classList.toggle("selected", c.dataset.sessionId === sessionId);
    });

    if (!els.expDetailPane) return;
    els.expDetailPane.innerHTML = `<div class="loading-state mono">Retrieving experiment telemetry & report…</div>`;

    try {
      const res = await fetch(`/api/experiments/${sessionId}`);
      if (!res.ok) throw new Error("Failed to load details");
      const data = await res.json();
      renderExperimentDetail(data);
    } catch (err) {
      els.expDetailPane.innerHTML = `<div class="empty-hint mono">ERROR LOADING EXPERIMENT DETAILS</div>`;
    }
  }

  function renderExperimentDetail(data) {
    if (!els.expDetailPane) return;

    const statusClass = data.is_current ? "active" : (data.status === "COMPLETED" ? "completed" : data.status === "RESOLVED" ? "resolved" : "hold");
    const container = document.createElement("div");

    // Hero Section
    const hero = el("div", "exp-hero");
    const heroTop = el("div", "exp-hero-top");
    const leftTitle = document.createElement("div");
    leftTitle.innerHTML = `
      <div class="exp-hero-id">${data.session_id}</div>
      <div class="exp-hero-sub mono">
        PROTOCOL: <b style="color:var(--cyan)">${data.protocol.toUpperCase()}</b> · CAMERA: ${data.camera} · STARTED: ${data.start_ts || "—"}
      </div>
    `;
    const rightActions = el("div", "exp-hero-actions mono");
    rightActions.innerHTML = `
      <span class="exp-status-pill ${statusClass}" style="font-size:11px;padding:3px 10px">${data.is_current ? "● LIVE ACTIVE" : data.status}</span>
      <a class="btn-dl" href="/api/experiments/${data.session_id}/report?download=true" target="_blank">📥 REPORT (.TXT)</a>
      <a class="btn-dl" href="/api/experiments/${data.session_id}/log?download=true" target="_blank">📥 AUDIT (.JSONL)</a>
      ${data.has_video ? `<a class="btn-dl sec" href="/api/experiments/${data.session_id}/video" target="_blank">🎥 VIDEO (${data.video_size_mb} MB)</a>` : ""}
    `;
    heroTop.appendChild(leftTitle);
    heroTop.appendChild(rightActions);
    hero.appendChild(heroTop);

    // Stats Grid
    const statsGrid = el("div", "exp-stats-grid mono");
    statsGrid.innerHTML = `
      <div class="exp-stat-item">
        <span class="exp-stat-label">TOTAL DURATION</span>
        <span class="exp-stat-val">${data.duration_str}</span>
      </div>
      <div class="exp-stat-item">
        <span class="exp-stat-label">VERIFIED STEPS</span>
        <span class="exp-stat-val" style="color:var(--green)">${data.steps_count} / ${data.total_protocol_steps}</span>
      </div>
      <div class="exp-stat-item">
        <span class="exp-stat-label">ANOMALIES / ALERTS</span>
        <span class="exp-stat-val" style="color:${data.anomalies_count > 0 ? "var(--amber)" : "var(--green)"}">${data.anomalies_count}</span>
      </div>
      <div class="exp-stat-item">
        <span class="exp-stat-label">AUDIT EVENTS</span>
        <span class="exp-stat-val">${data.events_count}</span>
      </div>
    `;
    hero.appendChild(statsGrid);
    container.appendChild(hero);

    // Tabs
    const tabsRow = el("div", "exp-tabs mono");
    tabsRow.innerHTML = `
      <button class="exp-tab-btn active" data-tab="steps">STEPS & ANOMALIES (${data.steps_count})</button>
      <button class="exp-tab-btn" data-tab="report">OFFICIAL REPORT (TXT)</button>
      <button class="exp-tab-btn" data-tab="audit">STRUCTURED AUDIT TRAIL</button>
    `;
    container.appendChild(tabsRow);

    // Tab content container
    const contentArea = el("div", "exp-tab-content");

    // Tab 1: Steps & Anomalies
    const tabSteps = el("div", "tab-pane-view");
    const stepsTitle = el("h3", "panel-title", "Verified Step Progression");
    stepsTitle.style.marginBottom = "10px";
    tabSteps.appendChild(stepsTitle);

    if (data.steps && data.steps.length > 0) {
      const stepCards = el("div", "step-cards-container");
      for (const s of data.steps) {
        const sc = el("div", "step-history-card");
        const shLeft = el("div", "sh-left");
        const shIcon = el("div", "sh-icon", "✔");
        const info = document.createElement("div");
        info.innerHTML = `
          <div class="sh-name">Step ${s.step}: ${s.name}</div>
          <div class="sh-id mono">${s.step_id} · ${s.ts ? s.ts.slice(11, 19) + "Z" : ""}</div>
        `;
        shLeft.appendChild(shIcon);
        shLeft.appendChild(info);

        const shMetrics = el("div", "sh-metrics");
        shMetrics.innerHTML = `
          <span class="sh-conf">CONF ${(s.confidence * 100).toFixed(1)}%</span>
          <span class="sh-dur">${s.duration_s}s</span>
        `;
        sc.appendChild(shLeft);
        sc.appendChild(shMetrics);
        stepCards.appendChild(sc);
      }
      tabSteps.appendChild(stepCards);
    } else {
      tabSteps.appendChild(el("div", "empty-hint mono", "NO STEP TRANSITIONS VERIFIED YET (EXPERIMENT IN PROGRESS)"));
    }

    // Anomalies section
    const anomTitle = el("h3", "panel-title", `Anomalies & Deviations (${data.anomalies_count})`);
    anomTitle.style.marginTop = "20px";
    anomTitle.style.marginBottom = "10px";
    tabSteps.appendChild(anomTitle);

    if (data.anomalies && data.anomalies.length > 0) {
      const table = el("table", "anomaly-table");
      table.innerHTML = `
        <thead>
          <tr>
            <th>TIME (UTC)</th>
            <th>SEVERITY</th>
            <th>CODE</th>
            <th>DETAIL / AUDIT EVENT</th>
          </tr>
        </thead>
        <tbody>
          ${data.anomalies.slice(0, 50).map((a) => `
            <tr>
              <td class="mono" style="color:var(--text-dim)">${a.ts ? a.ts.slice(11, 19) + "Z" : "—"}</td>
              <td><span class="anomaly-badge ${a.level}">${a.level ? a.level.toUpperCase() : "WARN"}</span></td>
              <td style="font-weight:bold;color:${a.level === "critical" ? "var(--red)" : "var(--amber)"}">${a.code || a.event}</td>
              <td style="color:var(--text)">${a.detail || "Anomaly recorded"}</td>
            </tr>
          `).join("")}
        </tbody>
      `;
      tabSteps.appendChild(table);
    } else {
      tabSteps.appendChild(el("div", "empty-hint mono", "ALL STEPS NOMINAL · NO DETECTED ANOMALIES"));
    }
    contentArea.appendChild(tabSteps);

    // Tab 2: Report View
    const tabReport = el("div", "tab-pane-view");
    tabReport.style.display = "none";
    const repActions = el("div", "exp-hero-actions mono");
    repActions.style.marginBottom = "8px";
    repActions.innerHTML = `
      <button class="btn-dl copy-rep-btn">📋 COPY REPORT TEXT</button>
      <a class="btn-dl" href="/api/experiments/${data.session_id}/report?download=true" target="_blank">💾 SAVE .TXT FILE</a>
    `;
    const repBox = el("pre", "raw-code-box", data.report_text || "No report generated.");
    repActions.querySelector(".copy-rep-btn").addEventListener("click", () => {
      navigator.clipboard.writeText(data.report_text || "");
      alert("Report text copied to clipboard.");
    });
    tabReport.appendChild(repActions);
    tabReport.appendChild(repBox);
    contentArea.appendChild(tabReport);

    // Tab 3: Raw Audit Trail View
    const tabAudit = el("div", "tab-pane-view");
    tabAudit.style.display = "none";
    const auditActions = el("div", "exp-hero-actions mono");
    auditActions.style.marginBottom = "8px";
    auditActions.innerHTML = `
      <button class="btn-dl copy-audit-btn">📋 COPY AUDIT LOG</button>
      <a class="btn-dl" href="/api/experiments/${data.session_id}/log?download=true" target="_blank">💾 SAVE .JSONL FILE</a>
    `;
    const auditBox = el("pre", "raw-code-box", "Loading audit stream…");
    tabAudit.appendChild(auditActions);
    tabAudit.appendChild(auditBox);
    contentArea.appendChild(tabAudit);

    container.appendChild(contentArea);

    // Tab switcher events
    tabsRow.querySelectorAll(".exp-tab-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        tabsRow.querySelectorAll(".exp-tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const tab = btn.dataset.tab;
        tabSteps.style.display = tab === "steps" ? "block" : "none";
        tabReport.style.display = tab === "report" ? "block" : "none";
        tabAudit.style.display = tab === "audit" ? "block" : "none";

        if (tab === "audit" && auditBox.textContent === "Loading audit stream…") {
          try {
            const r = await fetch(`/api/experiments/${data.session_id}/log`);
            const txt = await r.text();
            auditBox.textContent = txt || "(No audit events recorded)";
            auditActions.querySelector(".copy-audit-btn").addEventListener("click", () => {
              navigator.clipboard.writeText(txt);
              alert("Audit JSONL copied to clipboard.");
            });
          } catch {
            auditBox.textContent = "Error loading audit log.";
          }
        }
      });
    });

    els.expDetailPane.replaceChildren(container);
  }

  function wireExperimentsUI() {
    const openModal = () => {
      if (els.expModal) {
        els.expModal.classList.remove("hidden");
        loadExperiments(true);
      }
    };
    const closeModal = () => {
      if (els.expModal) {
        els.expModal.classList.add("hidden");
      }
    };

    if (els.btnOpenExperiments) els.btnOpenExperiments.addEventListener("click", openModal);
    if (els.btnToolbarExperiments) els.btnToolbarExperiments.addEventListener("click", openModal);
    if (els.btnCloseExperiments) els.btnCloseExperiments.addEventListener("click", closeModal);
    if (els.btnRefreshExperiments) els.btnRefreshExperiments.addEventListener("click", () => loadExperiments(true));

    if (els.expModal) {
      els.expModal.addEventListener("click", (e) => {
        if (e.target === els.expModal) closeModal();
      });
    }

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && els.expModal && !els.expModal.classList.contains("hidden")) {
        closeModal();
      }
    });

    if (els.expSearchInput) {
      els.expSearchInput.addEventListener("input", () => {
        renderExperimentList();
      });
    }

    document.querySelectorAll(".exp-filter-tags .filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".exp-filter-tags .filter-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeFilter = btn.dataset.filter;
        renderExperimentList();
      });
    });

    // Initial background count badge fetch
    loadExperiments(false);
  }

  /* ------------------------------------------------ boot */
  function boot() {
    wireVideoRecovery();
    connectTelemetry();
    loadCameras();
    wireExperimentsUI();
    els.camSelect.addEventListener("change", onCameraSelect);
    // Dismiss the boot overlay as soon as either channel proves alive.
    const dismiss = () => {
      els.boot.classList.add("gone");
      els.bootStatus.textContent = "Uplink established.";
    };
    els.video.addEventListener("load", dismiss, { once: true });
    setTimeout(dismiss, 4000); // never trap the user on the boot screen
  }

  boot();
})();
