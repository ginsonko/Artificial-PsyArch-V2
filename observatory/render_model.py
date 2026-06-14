from __future__ import annotations

import base64
import html
import io
import json
import math
import struct
import wave


def build_inner_world_render_model(observatory_payload: dict, *, inline_assets: bool = True) -> dict:
    """
    Convert APV2.1 observatory reconstruction into a UI-oriented render model.

    The reconstruction payload remains the white-box truth. This render model is
    a front-end contract: stable layer lists, canvas object geometry, and
    state-pool-derived audio synthesis. It intentionally does not expose raw
    image/audio assets as inner-world playback shortcuts.
    """

    inner_world = dict((observatory_payload or {}).get("inner_world", {}) or observatory_payload or {})
    vision = dict(inner_world.get("inner_vision_reconstruction", {}) or {})
    audio = dict(inner_world.get("inner_audio_reconstruction", {}) or {})
    layer_stack = _layer_stack(vision, audio)
    return {
        "schema_id": "apv21_inner_world_render_model/v1",
        "tick_index": (observatory_payload or {}).get("tick_index"),
        "layer_stack": layer_stack,
        "vision_panel": _build_vision_panel(vision, inline_assets=inline_assets),
        "audio_panel": _build_audio_panel(audio, inline_assets=inline_assets),
        "action_panel": dict((observatory_payload or {}).get("action", {}) or {}),
        "detail_panel": _build_detail_panel(observatory_payload, vision=vision, audio=audio),
        "layer_legend": _layer_legend(vision, audio),
    }


def render_inner_world_html(observatory_payload: dict, *, title: str = "APV2.1 Inner World", inline_assets: bool = True) -> str:
    model = build_inner_world_render_model(observatory_payload, inline_assets=inline_assets)
    data_json = json.dumps(model, ensure_ascii=False)
    safe_title = html.escape(str(title or "APV2.1 Inner World"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #17202a;
      --muted: #607080;
      --line: #d8e0e8;
      --real: #1f7a8c;
      --recalled: #8a5a18;
      --echo: #d05f45;
      --predicted: #6b5bd6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 20px; margin: 0 0 14px; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(320px, .8fr); gap: 16px; }}
    .panel {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .panel h2 {{ font-size: 15px; margin: 0 0 10px; }}
    canvas {{ width: 100%; aspect-ratio: 4 / 3; border: 1px solid var(--line); border-radius: 6px; background: #eef2f6; display: block; }}
    .audio-bars {{ display: flex; align-items: end; gap: 4px; height: 150px; padding: 8px; border: 1px solid var(--line); border-radius: 6px; background: #f3f6f8; }}
    .bar {{ flex: 1; min-width: 8px; border-radius: 4px 4px 0 0; background: var(--real); opacity: .78; }}
    .wave {{ margin-top: 10px; width: 100%; height: 84px; border: 1px solid var(--line); border-radius: 6px; }}
    audio {{ display: block; width: 100%; height: 36px; margin-top: 10px; }}
    .source-line {{ min-height: 20px; margin-top: 8px; color: var(--muted); }}
    .legend, .assets {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .chip {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; color: var(--muted); background: #fff; }}
    .real {{ border-color: color-mix(in srgb, var(--real) 55%, var(--line)); }}
    .recalled {{ border-color: color-mix(in srgb, var(--recalled) 55%, var(--line)); }}
    .echo {{ border-color: color-mix(in srgb, var(--echo) 55%, var(--line)); }}
    .predicted {{ border-color: color-mix(in srgb, var(--predicted) 55%, var(--line)); }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <div class="grid">
      <section class="panel">
        <h2>Vision</h2>
        <canvas id="visionCanvas" width="960" height="720"></canvas>
        <div id="visionAssets" class="assets"></div>
      </section>
      <section class="panel">
        <h2>Audio</h2>
        <div id="audioBars" class="audio-bars"></div>
        <canvas id="waveCanvas" class="wave" width="720" height="168"></canvas>
        <audio id="audioPlayer" controls preload="metadata"></audio>
        <div id="audioPlaybackSource" class="source-line"></div>
        <div id="audioAssets" class="assets"></div>
      </section>
    </div>
    <section class="panel" style="margin-top:16px">
      <h2>Layers</h2>
      <div id="legend" class="legend"></div>
    </section>
  </main>
  <script type="application/json" id="model">{html.escape(data_json)}</script>
  <script>
    const model = JSON.parse(document.getElementById('model').textContent);
    const colors = {{ real: '#1f7a8c', recalled: '#8a5a18', echo: '#d05f45', predicted: '#6b5bd6', proxy: '#d7e3ea' }};
    function chip(text, cls) {{
      const el = document.createElement('span');
      el.className = `chip ${{cls || ''}}`;
      el.textContent = text;
      return el;
    }}
    function drawVision() {{
      const canvas = document.getElementById('visionCanvas');
      const ctx = canvas.getContext('2d');
      const panel = model.vision_panel || {{}};
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const bg = panel.background || {{}};
      const rgb = bg.mean_rgb || [0.93, 0.95, 0.97];
      ctx.fillStyle = `rgb(${{Math.round(rgb[0]*255)}},${{Math.round(rgb[1]*255)}},${{Math.round(rgb[2]*255)}})`;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      drawField(ctx, canvas, panel.field || {{}});
      drawObjects(ctx, canvas, panel.objects || []);
      const assetBox = document.getElementById('visionAssets');
      assetBox.innerHTML = '';
      assetBox.appendChild(chip(panel.reconstruction_source || 'state_pool_numeric_channels', 'real'));
    }}
    function payloadValues(payload) {{
      return (payload && Array.isArray(payload.payload_values)) ? payload.payload_values : [];
    }}
    function drawField(ctx, canvas, field) {{
      const payloads = (field && field.payloads) || {{}};
      const color = payloads['vision.field.color_grid'];
      const shape = (color && color.payload_shape) || [];
      const values = payloadValues(color);
      if (shape.length < 3 || values.length < 3) return;
      const rows = shape[0] || 1;
      const cols = shape[1] || 1;
      const cw = canvas.width / cols;
      const ch = canvas.height / rows;
      for (let y = 0; y < rows; y++) {{
        for (let x = 0; x < cols; x++) {{
          const idx = (y * cols + x) * 3;
          const r = Math.round((values[idx] || 0) * 255);
          const g = Math.round((values[idx + 1] || 0) * 255);
          const b = Math.round((values[idx + 2] || 0) * 255);
          ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
          ctx.globalAlpha = 0.62;
          ctx.fillRect(x * cw, y * ch, Math.ceil(cw), Math.ceil(ch));
        }}
      }}
      ctx.globalAlpha = 1;
    }}
    function drawObjects(ctx, canvas, objects) {{
      objects.forEach(obj => {{
        const box = obj.bbox_norm || [0.5, 0.5, 0.3, 0.3];
        const x = (box[0] - box[2] / 2) * canvas.width;
        const y = (box[1] - box[3] / 2) * canvas.height;
        const w = box[2] * canvas.width;
        const h = box[3] * canvas.height;
        const rgb = obj.mean_rgb || [0.6, 0.7, 0.8];
        ctx.globalAlpha = Math.max(0.18, Math.min(0.96, obj.opacity || 0.5));
        drawObjectColorLayout(ctx, obj, x, y, w, h, rgb);
        ctx.globalAlpha = 1;
        ctx.lineWidth = obj.layer_type === 'recalled' ? 3 : 2;
        ctx.strokeStyle = colors[obj.layer_type] || colors.proxy;
        drawObjectContour(ctx, obj, x, y, w, h);
        ctx.strokeRect(x, y, w, h);
      }});
    }}
    function drawObjectColorLayout(ctx, obj, x, y, w, h, fallbackRgb) {{
      const color = obj.color_layout_payload || {{}};
      const mask = obj.mask_payload || {{}};
      const colorShape = color.payload_shape || [];
      const maskShape = mask.payload_shape || [];
      const colorValues = payloadValues(color);
      const maskValues = payloadValues(mask);
      if (colorShape.length < 3 || colorValues.length < 3) {{
        ctx.fillStyle = `rgb(${{Math.round(fallbackRgb[0]*255)}},${{Math.round(fallbackRgb[1]*255)}},${{Math.round(fallbackRgb[2]*255)}})`;
        ctx.fillRect(x, y, w, h);
        return;
      }}
      const rows = colorShape[0] || 1;
      const cols = colorShape[1] || 1;
      const mr = maskShape[0] || 0;
      const mc = maskShape[1] || 0;
      const cw = w / cols;
      const ch = h / rows;
      for (let row = 0; row < rows; row++) {{
        for (let col = 0; col < cols; col++) {{
          const idx = (row * cols + col) * 3;
          let alpha = 1;
          if (mr > 0 && mc > 0 && maskValues.length) {{
            const my = Math.min(mr - 1, Math.floor(row / rows * mr));
            const mx = Math.min(mc - 1, Math.floor(col / cols * mc));
            alpha = Math.max(0.08, Math.min(1, maskValues[my * mc + mx] || 0));
          }}
          ctx.globalAlpha = Math.max(0.08, Math.min(0.96, (obj.opacity || 0.5) * alpha));
          ctx.fillStyle = `rgb(${{Math.round((colorValues[idx] || 0) * 255)}},${{Math.round((colorValues[idx + 1] || 0) * 255)}},${{Math.round((colorValues[idx + 2] || 0) * 255)}})`;
          ctx.fillRect(x + col * cw, y + row * ch, Math.ceil(cw), Math.ceil(ch));
        }}
      }}
      ctx.globalAlpha = Math.max(0.18, Math.min(0.96, obj.opacity || 0.5));
      drawFocusDetailPatch(ctx, obj, x, y, w, h);
      ctx.globalAlpha = Math.max(0.18, Math.min(0.96, obj.opacity || 0.5));
    }}
    function drawFocusDetailPatch(ctx, obj, x, y, w, h) {{
      const patch = obj.focus_detail_patch_payload || {{}};
      const shape = patch.payload_shape || [];
      const values = payloadValues(patch);
      if (shape.length < 3 || values.length < 3) return;
      const rows = shape[0] || 1;
      const cols = shape[1] || 1;
      if (rows <= 0 || cols <= 0) return;
      const focus = obj.sampling_focus || {{}};
      const gain = Math.max(0, Math.min(1, focus.gain || obj.focus_precision || 0.75));
      const patchScale = Math.max(0.34, Math.min(0.92, 0.34 + gain * 0.5));
      const patchW = Math.max(2, w * patchScale);
      const patchH = Math.max(2, h * patchScale);
      const px = x + (w - patchW) * 0.5;
      const py = y + (h - patchH) * 0.5;
      const cw = patchW / cols;
      const ch = patchH / rows;
      ctx.save();
      ctx.globalAlpha = Math.max(0.18, Math.min(0.98, (obj.opacity || 0.5) * (0.74 + gain * 0.22)));
      for (let row = 0; row < rows; row++) {{
        for (let col = 0; col < cols; col++) {{
          const idx = (row * cols + col) * 3;
          ctx.fillStyle = `rgb(${{Math.round((values[idx] || 0) * 255)}},${{Math.round((values[idx + 1] || 0) * 255)}},${{Math.round((values[idx + 2] || 0) * 255)}})`;
          ctx.fillRect(px + col * cw, py + row * ch, Math.ceil(cw), Math.ceil(ch));
        }}
      }}
      ctx.strokeStyle = 'rgba(31,122,140,0.68)';
      ctx.lineWidth = 1;
      ctx.strokeRect(px, py, patchW, patchH);
      ctx.restore();
    }}
    function drawObjectContour(ctx, obj, x, y, w, h) {{
      const contour = obj.contour_payload || {{}};
      const values = payloadValues(contour);
      if (values.length < 4) return;
      ctx.beginPath();
      for (let idx = 0; idx + 1 < values.length; idx += 2) {{
        const px = x + (values[idx] || 0) * w;
        const py = y + (values[idx + 1] || 0) * h;
        if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }}
      ctx.closePath();
      ctx.stroke();
    }}
    function drawAudio() {{
      const panel = model.audio_panel || {{}};
      const bars = document.getElementById('audioBars');
      bars.innerHTML = '';
      (panel.spectrum_bands || []).forEach(row => {{
        const el = document.createElement('div');
        el.className = 'bar';
        el.style.height = `${{Math.max(4, Math.round((row.energy || 0) * 100))}}%`;
        el.title = row.label || '';
        bars.appendChild(el);
      }});
      const wave = document.getElementById('waveCanvas');
      const ctx = wave.getContext('2d');
      ctx.clearRect(0, 0, wave.width, wave.height);
      ctx.strokeStyle = colors.real;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const points = panel.waveform_preview || [];
      points.forEach((v, i) => {{
        const x = i / Math.max(1, points.length - 1) * wave.width;
        const y = wave.height * (0.5 - (v || 0) * 0.42);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      const player = document.getElementById('audioPlayer');
      const playback = panel.playback || {{}};
      const src = playback.source || {{}};
      const url = src.data_url || '';
      if (url) {{
        player.setAttribute('src', url);
        player.style.display = 'block';
      }} else {{
        player.removeAttribute('src');
        player.style.display = 'none';
      }}
      document.getElementById('audioPlaybackSource').textContent = playback.enabled
        ? `${{src.role || 'audio'}} · ${{panel.waveform_source || 'preview'}} · ${{Math.round(playback.duration_ms || 0)}} ms`
        : 'no playable audio';
      const assetBox = document.getElementById('audioAssets');
      assetBox.innerHTML = '';
      assetBox.appendChild(chip(panel.reconstruction_source || 'state_pool_numeric_channels', 'real'));
    }}
    function drawLegend() {{
      const legend = document.getElementById('legend');
      legend.innerHTML = '';
      (model.layer_legend || []).forEach(row => legend.appendChild(chip(`${{row.layer_type}} · ${{row.label}}`, row.layer_type)));
    }}
    drawVision(); drawAudio(); drawLegend();
  </script>
</body>
</html>
"""


def render_observatory_shell_html(*, title: str = "APV2.1 Native Observatory") -> str:
    """
    Return the lightweight localhost observatory shell.

    Unlike render_inner_world_html(), this page does not embed a single static
    model. It fetches the latest render model from the APV2.1-native server,
    which keeps the browser UI aligned with the realtime runtime and asset API.
    """

    safe_title = html.escape(str(title or "APV2.1 Native Observatory"))
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #16202a;
      --muted: #667789;
      --line: #d8e0e8;
      --real: #1f7a8c;
      --recalled: #8a5a18;
      --predicted: #6b5bd6;
      --echo: #d05f45;
      --soft: #edf2f6;
      --danger: #b64040;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { max-width: 1240px; margin: 0 auto; padding: 18px; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 14px;
      margin-bottom: 14px;
    }
    h1 { font-size: 20px; line-height: 1.2; margin: 0 0 5px; letter-spacing: 0; }
    h2 { font-size: 14px; line-height: 1.25; margin: 0 0 10px; letter-spacing: 0; }
    .status { min-height: 20px; color: var(--muted); }
    form { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
    input {
      width: min(44vw, 360px);
      min-width: 210px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
    }
    button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 12px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    button.primary { background: var(--ink); border-color: var(--ink); color: #fff; }
    button:disabled { opacity: .56; cursor: wait; }
    select {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 8px;
      background: #fff;
      color: var(--ink);
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(330px, .75fr);
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .wide { grid-column: 1 / -1; }
    canvas {
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
    }
    #visionCanvas { aspect-ratio: 4 / 3; }
    #waveCanvas { height: 88px; margin-top: 10px; }
    .audio-bars {
      display: flex;
      align-items: end;
      gap: 4px;
      height: 154px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
    }
    audio {
      display: block;
      width: 100%;
      height: 36px;
      margin-top: 10px;
    }
    .source-line {
      min-height: 20px;
      margin-top: 8px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .bar {
      flex: 1;
      min-width: 8px;
      border-radius: 4px 4px 0 0;
      background: var(--real);
      opacity: .82;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .chip {
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      background: #fff;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .real { border-color: color-mix(in srgb, var(--real) 55%, var(--line)); }
    .recalled { border-color: color-mix(in srgb, var(--recalled) 55%, var(--line)); }
    .echo { border-color: color-mix(in srgb, var(--echo) 55%, var(--line)); }
    .predicted { border-color: color-mix(in srgb, var(--predicted) 55%, var(--line)); }
    .error { color: var(--danger); }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .detail-col {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfd;
    }
    .detail-col h3 {
      margin: 0 0 6px;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
    }
    .detail-row {
      border-top: 1px solid var(--line);
      padding: 6px 0;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .detail-row:first-of-type { border-top: 0; }
    .detail-main { color: var(--ink); font-weight: 600; }
    .detail-meta { color: var(--muted); margin-top: 2px; }
    .timeline {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .timeline-row {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfd;
      min-width: 0;
    }
    .timeline-row strong {
      display: block;
      font-size: 12px;
      margin-bottom: 4px;
    }
    .timeline-row span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .playback-panel { margin-bottom: 14px; }
    .playback-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .playback-controls {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .frame-strip {
      display: flex;
      gap: 6px;
      overflow-x: auto;
      padding: 8px 0 2px;
      min-height: 42px;
    }
    .frame-button {
      flex: 0 0 auto;
      min-width: 62px;
      height: 30px;
      padding: 0 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .frame-button.active {
      border-color: var(--ink);
      background: var(--ink);
      color: #fff;
    }
    @media (max-width: 860px) {
      main { padding: 12px; }
      header { align-items: stretch; flex-direction: column; }
      form { justify-content: flex-start; }
      input { width: 100%; }
      .playback-bar { align-items: stretch; }
      .playback-controls { justify-content: flex-start; }
      .grid { grid-template-columns: 1fr; }
      .detail-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>__TITLE__</h1>
        <div id="status" class="status">Loading</div>
      </div>
      <form id="tickForm">
        <input id="tickText" aria-label="Tick text" value="inner world preview" autocomplete="off" />
        <button class="primary" id="runTick" type="submit">Run tick</button>
        <button id="refreshTick" type="button">Refresh</button>
      </form>
    </header>
    <section class="panel playback-panel">
      <div class="playback-bar">
        <div>
          <h2>Tick Playback</h2>
          <div id="playbackStatus" class="status">buffer loading</div>
        </div>
        <div class="playback-controls">
          <button id="capture8" type="button">Capture 8</button>
          <button id="refreshBuffer" type="button">Buffer</button>
          <button id="prevFrame" type="button">Prev</button>
          <button id="playPause" class="primary" type="button">Play</button>
          <button id="nextFrame" type="button">Next</button>
          <select id="playbackSpeed" aria-label="Playback speed">
            <option value="100">100 ms</option>
            <option value="250" selected>250 ms</option>
            <option value="500">500 ms</option>
          </select>
        </div>
      </div>
      <div id="frameStrip" class="frame-strip"></div>
    </section>
    <div class="grid">
      <section class="panel">
        <h2>Vision</h2>
        <canvas id="visionCanvas" width="960" height="720"></canvas>
        <div id="visionAssets" class="chips"></div>
      </section>
      <section class="panel">
        <h2>Audio</h2>
        <div id="audioBars" class="audio-bars"></div>
        <canvas id="waveCanvas" width="720" height="176"></canvas>
        <audio id="audioPlayer" controls preload="metadata"></audio>
        <div id="audioPlaybackSource" class="source-line"></div>
        <div id="audioAssets" class="chips"></div>
      </section>
      <section class="panel wide">
        <h2>Layers</h2>
        <div id="legend" class="chips"></div>
      </section>
      <section class="panel wide">
        <h2>Details</h2>
        <div id="detailChips" class="chips"></div>
        <div id="details" class="detail-grid"></div>
      </section>
      <section class="panel wide">
        <h2>Attention Timeline</h2>
        <div id="timelineChips" class="chips"></div>
        <div id="timeline" class="timeline"></div>
      </section>
    </div>
  </main>
  <script>
    const colors = { real: '#1f7a8c', recalled: '#8a5a18', echo: '#d05f45', predicted: '#6b5bd6', proxy: '#91a6b5' };
    const state = {
      model: null,
      busy: false,
      frames: [],
      frameIndex: -1,
      playing: false,
      playbackTimer: null,
      playbackIntervalMs: 250
    };

    function setStatus(text, error = false) {
      const el = document.getElementById('status');
      el.textContent = text;
      el.classList.toggle('error', Boolean(error));
    }

    function setBusy(value) {
      state.busy = Boolean(value);
      document.getElementById('runTick').disabled = state.busy;
      document.getElementById('refreshTick').disabled = state.busy;
      document.getElementById('capture8').disabled = state.busy;
      document.getElementById('refreshBuffer').disabled = state.busy;
    }

    function setPlaybackStatus(text, error = false) {
      const el = document.getElementById('playbackStatus');
      el.textContent = text;
      el.classList.toggle('error', Boolean(error));
    }

    function chip(text, cls) {
      const el = document.createElement('span');
      el.className = `chip ${cls || ''}`;
      el.textContent = text;
      return el;
    }

    async function fetchModel() {
      setBusy(true);
      try {
        const res = await fetch('/api/inner-world/render-model', { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.model = await res.json();
        window.__apv21Model = state.model;
        drawAll();
        setStatus(`tick ${state.model.tick_index ?? '-'}`);
        document.body.dataset.loaded = 'true';
        await fetchPlaybackBuffer({ selectLatest: true, redraw: false });
      } catch (err) {
        setStatus(`load failed: ${err.message}`, true);
      } finally {
        setBusy(false);
      }
    }

    async function runTick(text) {
      setBusy(true);
      try {
        const res = await fetch('/api/tick', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ text, use_demo_media: true, inline_assets: true })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        state.model = payload.render_model;
        window.__apv21Model = state.model;
        drawAll();
        setStatus(`tick ${state.model.tick_index ?? '-'}`);
        await fetchPlaybackBuffer({ selectLatest: true, redraw: false });
      } catch (err) {
        setStatus(`tick failed: ${err.message}`, true);
      } finally {
        setBusy(false);
      }
    }

    async function fetchPlaybackBuffer(options = {}) {
      const selectLatest = options.selectLatest !== false;
      const redraw = options.redraw !== false;
      try {
        const res = await fetch('/api/inner-world/playback-buffer', { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        state.frames = Array.isArray(payload.frames) ? payload.frames : [];
        const latestIndex = state.frames.length - 1;
        if (selectLatest || state.frameIndex < 0 || state.frameIndex >= state.frames.length) {
          state.frameIndex = latestIndex;
        }
        if (state.frameIndex >= 0 && state.frames[state.frameIndex]) {
          const model = state.frames[state.frameIndex].render_model || {};
          if (redraw) {
            state.model = model;
            window.__apv21Model = state.model;
            drawAll();
            setStatus(`tick ${state.model.tick_index ?? '-'}`);
          }
        }
        drawFrameStrip();
        const tick = state.frames[state.frameIndex] ? state.frames[state.frameIndex].tick_index : '-';
        setPlaybackStatus(`${state.frames.length}/${payload.max_frames || '-'} frames · selected tick ${tick}`);
        window.__apv21PlaybackBuffer = payload;
        return payload;
      } catch (err) {
        setPlaybackStatus(`buffer failed: ${err.message}`, true);
        return null;
      }
    }

    async function captureTicks(count) {
      stopPlayback();
      setBusy(true);
      try {
        const text = document.getElementById('tickText').value || 'inner world preview';
        const total = Math.max(1, Math.min(16, Number(count) || 8));
        for (let idx = 0; idx < total; idx += 1) {
          setStatus(`capturing ${idx + 1}/${total}`);
          const res = await fetch('/api/tick', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ text, use_demo_media: true, inline_assets: true })
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const payload = await res.json();
          state.model = payload.render_model;
          window.__apv21Model = state.model;
        }
        await fetchPlaybackBuffer({ selectLatest: true, redraw: true });
        setStatus(`captured ${total} ticks · tick ${state.model && state.model.tick_index !== undefined ? state.model.tick_index : '-'}`);
      } catch (err) {
        setStatus(`capture failed: ${err.message}`, true);
      } finally {
        setBusy(false);
      }
    }

    function selectFrame(index, options = {}) {
      if (!state.frames.length) return;
      const bounded = Math.max(0, Math.min(state.frames.length - 1, Number(index) || 0));
      state.frameIndex = bounded;
      const frame = state.frames[state.frameIndex] || {};
      state.model = frame.render_model || {};
      window.__apv21Model = state.model;
      window.__apv21SelectedFrame = frame;
      drawAll();
      drawFrameStrip();
      const delta = frame.gaze_delta || {};
      const dxdy = Array.isArray(delta.delta_norm) && delta.delta_norm.length >= 2
        ? ` · gaze Δ ${formatNorm(delta.delta_norm[0])},${formatNorm(delta.delta_norm[1])}`
        : '';
      setStatus(`tick ${state.model.tick_index ?? '-'}${dxdy}`);
      setPlaybackStatus(`${state.frameIndex + 1}/${state.frames.length} frames · tick ${frame.tick_index ?? '-'}${dxdy}`);
      if (options.playAudio) playCurrentFrameAudio();
    }

    function drawFrameStrip() {
      const strip = document.getElementById('frameStrip');
      strip.innerHTML = '';
      state.frames.forEach((frame, index) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `frame-button ${index === state.frameIndex ? 'active' : ''}`;
        btn.textContent = `t${frame.tick_index ?? '-'}`;
        const delta = frame.gaze_delta || {};
        if (Array.isArray(delta.delta_norm) && delta.delta_norm.length >= 2) {
          btn.title = `frame ${frame.frame_seq || index + 1} · gaze delta ${formatNorm(delta.delta_norm[0])},${formatNorm(delta.delta_norm[1])} · ${delta.action_id || ''}`;
        } else {
          btn.title = `frame ${frame.frame_seq || index + 1}`;
        }
        btn.addEventListener('click', () => {
          stopPlayback();
          selectFrame(index);
        });
        strip.appendChild(btn);
      });
    }

    function stepFrame(direction, options = {}) {
      if (!state.frames.length) return;
      const next = (state.frameIndex + direction + state.frames.length) % state.frames.length;
      selectFrame(next, options);
    }

    function startPlayback() {
      if (!state.frames.length) return;
      if (state.playing) return;
      state.playing = true;
      document.getElementById('playPause').textContent = 'Pause';
      selectFrame(state.frameIndex < 0 ? 0 : state.frameIndex, { playAudio: true });
      state.playbackTimer = window.setInterval(() => {
        stepFrame(1, { playAudio: true });
      }, Math.max(60, Number(state.playbackIntervalMs) || 250));
    }

    function stopPlayback() {
      state.playing = false;
      if (state.playbackTimer) {
        window.clearInterval(state.playbackTimer);
        state.playbackTimer = null;
      }
      const button = document.getElementById('playPause');
      if (button) button.textContent = 'Play';
    }

    function togglePlayback() {
      if (state.playing) stopPlayback(); else startPlayback();
    }

    function playCurrentFrameAudio() {
      const player = document.getElementById('audioPlayer');
      const panel = (state.model && state.model.audio_panel) || {};
      const playback = panel.playback || {};
      const src = playback.source || {};
      const url = (src && src.data_url) || '';
      if (!url) return;
      if (player.getAttribute('src') !== url) player.setAttribute('src', url);
      try {
        player.currentTime = 0;
        const promise = player.play();
        if (promise && typeof promise.catch === 'function') {
          promise.catch(() => setPlaybackStatus('visual playback running · click audio control if sound is blocked'));
        }
      } catch (err) {
        setPlaybackStatus('visual playback running · audio play blocked');
      }
    }

    function drawAll() {
      drawVision();
      drawAudio();
      drawLegend();
      drawDetails();
      drawTimeline();
    }

    function drawVision() {
      const canvas = document.getElementById('visionCanvas');
      const ctx = canvas.getContext('2d');
      const panel = (state.model && state.model.vision_panel) || {};
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const bg = panel.background || {};
      const rgb = bg.mean_rgb || [0.93, 0.95, 0.97];
      ctx.fillStyle = `rgb(${Math.round(rgb[0] * 255)},${Math.round(rgb[1] * 255)},${Math.round(rgb[2] * 255)})`;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      drawField(ctx, canvas, panel.field || {});
      drawObjects(ctx, canvas, panel.objects || []);
      drawFocusOverlay(ctx, canvas, panel.focus_overlay || {});
      window.__apv21VisionDrawn = true;
      const assetBox = document.getElementById('visionAssets');
      assetBox.innerHTML = '';
      assetBox.appendChild(chip(panel.reconstruction_source || 'state_pool_numeric_channels', 'real'));
      const focus = panel.focus_overlay || {};
      if (focus.visible) {
        const c = focus.clarity || {};
        assetBox.appendChild(chip(`gaze ${formatNorm((focus.center_norm || [0.5, 0.5])[0])},${formatNorm((focus.center_norm || [0.5, 0.5])[1])}`, 'real'));
        assetBox.appendChild(chip(`clarity ${formatNorm(c.near_focus || 0)} / ${formatNorm(c.far_periphery || 0)}`, 'real'));
        const r = focus.resolution_summary || {};
        if (r.variable_resolution_active) assetBox.appendChild(chip(`foveated res ${r.max_color_cells || 0}/${r.min_color_cells || 0}`, 'real'));
      }
    }

    function payloadValues(payload) {
      return (payload && Array.isArray(payload.payload_values)) ? payload.payload_values : [];
    }

    function drawField(ctx, canvas, field) {
      const payloads = (field && field.payloads) || {};
      const color = payloads['vision.field.color_grid'];
      const shape = (color && color.payload_shape) || [];
      const values = payloadValues(color);
      if (shape.length < 3 || values.length < 3) return;
      const rows = shape[0] || 1;
      const cols = shape[1] || 1;
      const cw = canvas.width / cols;
      const ch = canvas.height / rows;
      ctx.globalAlpha = 0.62;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const idx = (y * cols + x) * 3;
          ctx.fillStyle = `rgb(${Math.round((values[idx] || 0) * 255)},${Math.round((values[idx + 1] || 0) * 255)},${Math.round((values[idx + 2] || 0) * 255)})`;
          ctx.fillRect(x * cw, y * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      ctx.globalAlpha = 1;
    }

    function drawFocusOverlay(ctx, canvas, focus) {
      if (!focus || !focus.visible) return;
      const center = Array.isArray(focus.center_norm) ? focus.center_norm : [0.5, 0.5];
      const cx = Math.max(0, Math.min(1, center[0] ?? 0.5)) * canvas.width;
      const cy = Math.max(0, Math.min(1, center[1] ?? 0.5)) * canvas.height;
      const radiusNorm = Math.max(0.04, Math.min(0.9, focus.radius_norm || 0.42));
      const radius = radiusNorm * Math.min(canvas.width, canvas.height);
      drawPrecisionHeatmap(ctx, canvas, focus.precision_grid || {});
      ctx.save();
      ctx.fillStyle = 'rgba(246,248,251,0.32)';
      ctx.beginPath();
      ctx.rect(0, 0, canvas.width, canvas.height);
      ctx.arc(cx, cy, radius, 0, Math.PI * 2, true);
      ctx.fill('evenodd');
      const grad = ctx.createRadialGradient(cx, cy, radius * 0.52, cx, cy, radius * 1.24);
      grad.addColorStop(0, 'rgba(255,255,255,0)');
      grad.addColorStop(1, 'rgba(246,248,251,0.42)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'rgba(22,32,42,0.82)';
      ctx.lineWidth = 2;
      ctx.setLineDash([9, 6]);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = 'rgba(31,122,140,0.95)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - 16, cy);
      ctx.lineTo(cx + 16, cy);
      ctx.moveTo(cx, cy - 16);
      ctx.lineTo(cx, cy + 16);
      ctx.stroke();
      ctx.restore();
    }

    function drawPrecisionHeatmap(ctx, canvas, precisionGrid) {
      const values = payloadValues(precisionGrid);
      const shape = (precisionGrid && precisionGrid.payload_shape) || [];
      if (shape.length < 2 || values.length === 0) return;
      const rows = shape[0] || 1;
      const cols = shape[1] || 1;
      const cw = canvas.width / cols;
      const ch = canvas.height / rows;
      ctx.save();
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const value = Math.max(0, Math.min(1, values[y * cols + x] || 0));
          ctx.globalAlpha = 0.05 + value * 0.16;
          ctx.fillStyle = `rgb(${Math.round(255 - value * 80)},${Math.round(242 + value * 8)},${Math.round(168 + value * 72)})`;
          ctx.fillRect(x * cw, y * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      ctx.restore();
    }

    function drawObjects(ctx, canvas, objects) {
      const rows = [...objects].sort((a, b) => (a.z_index || 0) - (b.z_index || 0));
      rows.forEach(obj => {
        const box = Array.isArray(obj.bbox_norm) && obj.bbox_norm.length >= 4 ? obj.bbox_norm : [0.5, 0.5, 0.3, 0.3];
        const x = (box[0] - box[2] / 2) * canvas.width;
        const y = (box[1] - box[3] / 2) * canvas.height;
        const w = Math.max(2, box[2] * canvas.width);
        const h = Math.max(2, box[3] * canvas.height);
        const rgb = Array.isArray(obj.mean_rgb) && obj.mean_rgb.length >= 3 ? obj.mean_rgb : [0.58, 0.66, 0.72];
        ctx.globalAlpha = Math.max(0.2, Math.min(0.96, obj.opacity || 0.46));
        drawObjectColorLayout(ctx, obj, x, y, w, h, rgb);
        ctx.globalAlpha = 1;
        ctx.lineWidth = obj.layer_type === 'recalled' ? 3 : 2;
        ctx.strokeStyle = colors[obj.layer_type] || colors.proxy;
        drawObjectContour(ctx, obj, x, y, w, h);
        ctx.strokeRect(x, y, w, h);
      });
    }

    function drawObjectColorLayout(ctx, obj, x, y, w, h, fallbackRgb) {
      const color = obj.color_layout_payload || {};
      const mask = obj.mask_payload || {};
      const colorShape = color.payload_shape || [];
      const maskShape = mask.payload_shape || [];
      const colorValues = payloadValues(color);
      const maskValues = payloadValues(mask);
      if (colorShape.length < 3 || colorValues.length < 3) {
        ctx.fillStyle = `rgb(${Math.round(fallbackRgb[0] * 255)},${Math.round(fallbackRgb[1] * 255)},${Math.round(fallbackRgb[2] * 255)})`;
        ctx.fillRect(x, y, w, h);
        return;
      }
      const rows = colorShape[0] || 1;
      const cols = colorShape[1] || 1;
      const mr = maskShape[0] || 0;
      const mc = maskShape[1] || 0;
      const cw = w / cols;
      const ch = h / rows;
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const idx = (row * cols + col) * 3;
          let alpha = 1;
          if (mr > 0 && mc > 0 && maskValues.length) {
            const my = Math.min(mr - 1, Math.floor(row / rows * mr));
            const mx = Math.min(mc - 1, Math.floor(col / cols * mc));
            alpha = Math.max(0.08, Math.min(1, maskValues[my * mc + mx] || 0));
          }
          ctx.globalAlpha = Math.max(0.08, Math.min(0.96, (obj.opacity || 0.46) * alpha));
          ctx.fillStyle = `rgb(${Math.round((colorValues[idx] || 0) * 255)},${Math.round((colorValues[idx + 1] || 0) * 255)},${Math.round((colorValues[idx + 2] || 0) * 255)})`;
          ctx.fillRect(x + col * cw, y + row * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      ctx.globalAlpha = Math.max(0.2, Math.min(0.96, obj.opacity || 0.46));
      drawFocusDetailPatch(ctx, obj, x, y, w, h);
      ctx.globalAlpha = Math.max(0.2, Math.min(0.96, obj.opacity || 0.46));
    }

    function drawFocusDetailPatch(ctx, obj, x, y, w, h) {
      const patch = obj.focus_detail_patch_payload || {};
      const shape = patch.payload_shape || [];
      const values = payloadValues(patch);
      if (shape.length < 3 || values.length < 3) return;
      const rows = shape[0] || 1;
      const cols = shape[1] || 1;
      if (rows <= 0 || cols <= 0) return;
      const focus = obj.sampling_focus || {};
      const gain = Math.max(0, Math.min(1, focus.gain || obj.focus_precision || 0.75));
      const patchScale = Math.max(0.34, Math.min(0.92, 0.34 + gain * 0.5));
      const patchW = Math.max(2, w * patchScale);
      const patchH = Math.max(2, h * patchScale);
      const px = x + (w - patchW) * 0.5;
      const py = y + (h - patchH) * 0.5;
      const cw = patchW / cols;
      const ch = patchH / rows;
      ctx.save();
      ctx.globalAlpha = Math.max(0.18, Math.min(0.98, (obj.opacity || 0.46) * (0.74 + gain * 0.22)));
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const idx = (row * cols + col) * 3;
          ctx.fillStyle = `rgb(${Math.round((values[idx] || 0) * 255)},${Math.round((values[idx + 1] || 0) * 255)},${Math.round((values[idx + 2] || 0) * 255)})`;
          ctx.fillRect(px + col * cw, py + row * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      ctx.strokeStyle = 'rgba(31,122,140,0.68)';
      ctx.lineWidth = 1;
      ctx.strokeRect(px, py, patchW, patchH);
      ctx.restore();
    }

    function drawObjectContour(ctx, obj, x, y, w, h) {
      const contour = obj.contour_payload || {};
      const values = payloadValues(contour);
      if (values.length < 4) return;
      ctx.beginPath();
      for (let idx = 0; idx + 1 < values.length; idx += 2) {
        const px = x + (values[idx] || 0) * w;
        const py = y + (values[idx + 1] || 0) * h;
        if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
    }

    function drawAudio() {
      const panel = (state.model && state.model.audio_panel) || {};
      const bars = document.getElementById('audioBars');
      bars.innerHTML = '';
      (panel.spectrum_bands || []).forEach(row => {
        const el = document.createElement('div');
        el.className = 'bar';
        el.style.height = `${Math.max(4, Math.round((row.energy || 0) * 100))}%`;
        const focusGain = Math.max(0, Math.min(1, row.focus_gain || 0));
        el.style.background = focusGain > 0.24 ? '#6b5bd6' : colors.real;
        el.style.opacity = `${0.55 + focusGain * 0.42}`;
        el.style.boxShadow = focusGain > 0.6 ? '0 0 0 2px rgba(107,91,214,.25)' : 'none';
        el.title = row.label || '';
        bars.appendChild(el);
      });
      const wave = document.getElementById('waveCanvas');
      const ctx = wave.getContext('2d');
      ctx.clearRect(0, 0, wave.width, wave.height);
      ctx.fillStyle = '#edf2f6';
      ctx.fillRect(0, 0, wave.width, wave.height);
      ctx.strokeStyle = colors.real;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const points = panel.waveform_preview || [];
      points.forEach((v, i) => {
        const x = i / Math.max(1, points.length - 1) * wave.width;
        const y = wave.height * (0.5 - (v || 0) * 0.42);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      window.__apv21AudioDrawn = true;
      const player = document.getElementById('audioPlayer');
      const playback = panel.playback || {};
      const src = playback.source || {};
      const url = (src && src.data_url) || '';
      if (url) {
        if (player.getAttribute('src') !== url) player.setAttribute('src', url);
        player.style.display = 'block';
      } else {
        player.removeAttribute('src');
        player.style.display = 'none';
      }
      document.getElementById('audioPlaybackSource').textContent = playback.enabled
        ? `${src.role || 'audio'} · ${panel.waveform_source || 'preview'} · ${Math.round(playback.duration_ms || 0)} ms`
        : 'no playable audio';
      const assetBox = document.getElementById('audioAssets');
      assetBox.innerHTML = '';
      assetBox.appendChild(chip(panel.reconstruction_source || 'state_pool_numeric_channels', 'real'));
      const band = panel.focus_band_overlay || {};
      if (band.visible) {
        assetBox.appendChild(chip(`focus ${Math.round(band.center_hz || 0)} Hz`, 'predicted'));
        assetBox.appendChild(chip(`band ${Math.round(band.width_hz || 0)} Hz · p ${formatNorm(band.precision || 0)}`, 'predicted'));
      }
    }

    function drawLegend() {
      const legend = document.getElementById('legend');
      legend.innerHTML = '';
      ((state.model && state.model.layer_legend) || []).forEach(row => {
        legend.appendChild(chip(`${row.source} · ${row.layer_type} · ${row.label}`, row.layer_type));
      });
    }

    function drawDetails() {
      const panel = (state.model && state.model.detail_panel) || {};
      const chips = document.getElementById('detailChips');
      const details = document.getElementById('details');
      chips.innerHTML = '';
      details.innerHTML = '';
      const focus = panel.focus_summary || {};
      if (focus.center_norm) chips.appendChild(chip(`focus ${formatNorm(focus.center_norm[0])},${formatNorm(focus.center_norm[1])}`, 'real'));
      if (focus.resolution_summary) {
        const r = focus.resolution_summary;
        chips.appendChild(chip(`resolution ${r.max_color_cells || 0}/${r.min_color_cells || 0}`, 'real'));
        if (r.focus_detail_patch_count) chips.appendChild(chip(`fovea patch ${r.focus_detail_patch_count} 路 ${r.max_focus_patch_cells || 0}`, 'real'));
      }
      const groups = [
        ['fast_bn', 'Fast Bn', 'recalled'],
        ['slow_bn_prime', \"Slow Bn'\", 'recalled'],
        ['fast_cn', 'Fast Cn', 'predicted'],
        ['slow_cn_prime', \"Slow Cn'\", 'predicted'],
      ];
      groups.forEach(([key, title, cls]) => {
        const col = document.createElement('div');
        col.className = 'detail-col';
        const h = document.createElement('h3');
        h.textContent = title;
        col.appendChild(h);
        const rows = (panel.memory_layers && panel.memory_layers[key]) || [];
        if (!rows.length) {
          const empty = document.createElement('div');
          empty.className = 'detail-row detail-meta';
          empty.textContent = 'no active row';
          col.appendChild(empty);
        }
        rows.slice(0, 4).forEach(row => {
          const item = document.createElement('div');
          item.className = `detail-row ${cls}`;
          const main = document.createElement('div');
          main.className = 'detail-main';
          main.textContent = row.memory_id || row.successor_memory_id || row.layer_id || 'row';
          const meta = document.createElement('div');
          meta.className = 'detail-meta';
          meta.textContent = `score ${formatNorm(row.score || row.bn_score || 0)} · energy ${formatNorm(row.virtual_energy_sum || row.energy_total || 0)} · ${(row.top_labels || row.predicted_labels || []).slice(0, 3).join(', ')}`;
          item.appendChild(main);
          item.appendChild(meta);
          col.appendChild(item);
        });
        details.appendChild(col);
      });
    }

    function drawTimeline() {
      const rows = ((state.model && state.model.attention_timeline) || []).slice(-10);
      const chips = document.getElementById('timelineChips');
      const box = document.getElementById('timeline');
      chips.innerHTML = '';
      box.innerHTML = '';
      chips.appendChild(chip(`${rows.length} tick window`, 'real'));
      rows.forEach(row => {
        const item = document.createElement('div');
        item.className = 'timeline-row';
        const c = row.clarity || {};
        const r = row.resolution || {};
        const gaze = Array.isArray(row.gaze_center_norm) && row.gaze_center_norm.length >= 2
          ? `${formatNorm(row.gaze_center_norm[0])},${formatNorm(row.gaze_center_norm[1])}`
          : '-';
        item.innerHTML = '';
        const title = document.createElement('strong');
        title.textContent = `tick ${row.tick_index ?? '-'}`;
        const line1 = document.createElement('span');
        line1.textContent = `gaze ${gaze} 路 clarity ${formatNorm(c.near_focus || 0)}/${formatNorm(c.far_periphery || 0)}`;
        const line2 = document.createElement('span');
        line2.textContent = `res ${r.max_color_cells || 0}/${r.min_color_cells || 0} 路 patch ${r.focus_detail_patch_count || 0}`;
        const line3 = document.createElement('span');
        line3.textContent = `B ${joinShort(row.fast_bn_labels)} | B' ${joinShort(row.slow_bn_labels)}`;
        const line4 = document.createElement('span');
        line4.textContent = `C ${joinShort(row.fast_cn_labels)} | C' ${joinShort(row.slow_cn_labels)}`;
        const gazeAction = row.gaze_action || {};
        const line5 = document.createElement('span');
        const target = gazeAction.target || '-';
        const reason = gazeAction.reason || gazeAction.action_id || '-';
        const learned = gazeAction.parameter_memory_bias ? ` 路 learned ${formatNorm(gazeAction.parameter_drive_bias || 0)}` : '';
        line5.textContent = gazeAction.selected
          ? `gaze action ${gazeAction.action_id || '-'} -> ${target} 路 ${reason} 路 move ${formatNorm(gazeAction.movement_distance || 0)} 路 fatigue ${formatNorm(gazeAction.target_fatigue || 0)}${learned}`
          : 'gaze action -';
        item.appendChild(title);
        item.appendChild(line1);
        item.appendChild(line2);
        item.appendChild(line3);
        item.appendChild(line4);
        item.appendChild(line5);
        box.appendChild(item);
      });
    }

    function formatNorm(value) {
      return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : '0.00';
    }

    function joinShort(values) {
      const rows = Array.isArray(values) ? values.filter(Boolean).slice(0, 3) : [];
      return rows.length ? rows.join(', ') : '-';
    }

    document.getElementById('tickForm').addEventListener('submit', event => {
      event.preventDefault();
      runTick(document.getElementById('tickText').value || 'inner world preview');
    });
    document.getElementById('refreshTick').addEventListener('click', fetchModel);
    document.getElementById('refreshBuffer').addEventListener('click', () => fetchPlaybackBuffer({ selectLatest: true, redraw: true }));
    document.getElementById('capture8').addEventListener('click', () => captureTicks(8));
    document.getElementById('prevFrame').addEventListener('click', () => {
      stopPlayback();
      stepFrame(-1);
    });
    document.getElementById('nextFrame').addEventListener('click', () => {
      stopPlayback();
      stepFrame(1);
    });
    document.getElementById('playPause').addEventListener('click', togglePlayback);
    document.getElementById('playbackSpeed').addEventListener('change', event => {
      state.playbackIntervalMs = Number(event.target.value) || 250;
      if (state.playing) {
        stopPlayback();
        startPlayback();
      }
    });
    fetchModel();
  </script>
</body>
</html>
""".replace("__TITLE__", safe_title)


def _build_vision_panel(vision: dict, *, inline_assets: bool) -> dict:
    objects = []
    for row in list(vision.get("objects", []) or []):
        if not isinstance(row, dict):
            continue
        objects.append(
            {
                "slot": int(row.get("slot", len(objects)) or len(objects)),
                "layer_type": _render_layer_type(str(row.get("source", "real_input") or "real_input")),
                "bbox_norm": list(row.get("bbox_norm", []) or []),
                "mean_rgb": list(row.get("mean_rgb", []) or []),
                "motion_vector": list(row.get("motion_vector", []) or []),
                "palette": list(row.get("palette", []) or []),
                "mask_payload": _render_payload(row.get("mask_payload", {})),
                "contour_payload": _render_payload(row.get("contour_payload", {})),
                "color_layout_payload": _render_payload(row.get("color_layout_payload", {})),
                "edge_layout_payload": _render_payload(row.get("edge_layout_payload", {})),
                # A peripheral or recalled object may legitimately have no fovea
                # patch. The UI contract still keeps the payload shape stable so
                # "no high-detail sample here" is explicit instead of becoming a
                # missing-key failure in playback/tests.
                "focus_detail_patch_payload": _render_payload(row.get("focus_detail_patch_payload", {})),
                "opacity": float(row.get("opacity", 0.4) or 0.4),
                "z_index": int(row.get("z_index", 0) or 0),
                "asset_id": "",
                "focus_tile_asset_id": "",
                "reconstruction_basis": str(row.get("reconstruction_basis", "") or "state_pool_numeric_channels"),
                "sampling_focus": dict(row.get("sampling_focus", {}) or {}),
                "focus_precision": float(row.get("focus_precision", 0.0) or (row.get("sampling_focus", {}) or {}).get("precision", 0.0) or 0.0)
                if isinstance(row.get("sampling_focus", {}) or {}, dict)
                else float(row.get("focus_precision", 0.0) or 0.0),
                "variable_resolution": dict(row.get("variable_resolution", {}) or {}),
                "not_new_external_input": bool(row.get("not_new_external_input", False)),
                "echo_kind": str(row.get("echo_kind", "") or ""),
                "echo_modality": str(row.get("echo_modality", "") or ""),
                "age_ticks": int(row.get("age_ticks", 0) or 0),
            }
        )
    objects.sort(key=lambda row: (int(row.get("z_index", 0) or 0), str(row.get("layer_type", ""))))
    field = dict(vision.get("field", {}) or {})
    focus_overlay = dict(vision.get("focus_overlay", {}) or {})
    return {
        "schema_id": "apv21_vision_render_panel/v1",
        "reconstruction_source": "state_pool_numeric_channels",
        "asset_shortcut_used": False,
        "field": {
            "schema_id": "vision_field_render/v1",
            "payloads": dict(field.get("payloads", {}) or {}),
        },
        "focus_overlay": focus_overlay,
        "background": _vision_background(objects),
        "layers": list(vision.get("layers", []) or []),
        "objects": objects,
        "image_sources": [],
        "predicted_labels": list(vision.get("predicted_labels", []) or []),
    }


def _render_payload(payload: object) -> dict:
    """
    Normalize numeric reconstruction payloads for the browser contract.

    Cognition is allowed to omit a channel when AP did not sample it, especially
    for foveated detail outside the current gaze. The observatory is only a
    read-only view, so it should express that absence as an empty numeric
    payload rather than changing the object schema from tick to tick.
    """

    row = dict(payload) if isinstance(payload, dict) else {}
    if not isinstance(row.get("payload_shape", []), (list, tuple)):
        row["payload_shape"] = []
    if not isinstance(row.get("payload_values", []), (list, tuple)):
        row["payload_values"] = []
    row.setdefault("payload_shape", [])
    row.setdefault("payload_values", [])
    return row


def _build_audio_panel(audio: dict, *, inline_assets: bool) -> dict:
    events = [row for row in list(audio.get("events", []) or []) if isinstance(row, dict)]
    event = (events or [{}])[0]
    feature_summary = dict(event.get("feature_summary", {}) or {})
    band_labels = list(event.get("current_bands", []) or [])
    rms = max(0.0, float(feature_summary.get("rms", 0.0) or 0.0))
    band_count = max(8, len(band_labels), 12)
    spectral_hint = float(feature_summary.get("spectral_centroid_hz", 0.0) or 0.0)
    spectrum = _spectrum_from_audio_payload(event, band_count=band_count)
    if not spectrum:
        spectrum = _synthetic_spectrum(band_count=band_count, centroid=spectral_hint, rms=rms)
    waveform_preview = _waveform_preview_from_audio_payload(event, points=128)
    if not waveform_preview:
        waveform_preview = _synthetic_waveform(
            rms=rms,
            dominant_hz=float(feature_summary.get("dominant_hz", 0.0) or 0.0),
            onset=float(feature_summary.get("onset_strength", 0.0) or 0.0),
            points=128,
        )
    sample_rate = max(8000, int(((inner_preview := dict(audio.get("preview", {}) or {})).get("sample_rate", 0) or feature_summary.get("sample_rate", 0) or 16000)))
    duration_ms = max(220.0, min(1600.0, float(inner_preview.get("preview_duration_ms", 0.0) or 420.0)))
    mix_trace = _audio_event_mix_trace(events)
    playback_allowed = _audio_playback_allowed(events, preview=inner_preview)
    playback_source = _synthesized_audio_source(
        event,
        waveform_preview,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        events=events,
        mix_trace=mix_trace,
    ) if playback_allowed["allowed"] else {}
    focus_band_overlay = dict(audio.get("focus_band_overlay", {}) or (audio.get("focus", {}) or {}).get("focus_band_overlay", {}) or {})
    diagnostics = _audio_reconstruction_diagnostics(
        event=event,
        preview=inner_preview,
        playback_source=playback_source,
        duration_ms=duration_ms,
        mix_trace=mix_trace,
    )
    return {
        "schema_id": "apv21_audio_render_panel/v1",
        "reconstruction_source": "state_pool_numeric_channels",
        "asset_shortcut_used": False,
        "layers": list(audio.get("layers", []) or []),
        "spectrum_bands": [
            {
                "label": band_labels[idx] if idx < len(band_labels) else f"band_{idx}",
                "energy": value,
                "focus_gain": _audio_band_focus_gain(idx=idx, band_count=len(spectrum), focus_band_overlay=focus_band_overlay),
            }
            for idx, value in enumerate(spectrum)
        ],
        "waveform_preview": waveform_preview,
        "waveform_source": "state_pool_numeric_synthesis",
        "spectrum_source": "state_pool_numeric_channels",
        "focus_band_overlay": focus_band_overlay,
        "playback": {
            "enabled": bool(playback_source),
            "source": playback_source,
            "duration_ms": round(duration_ms, 3),
            "decoded_for_preview": False,
            "synthesis_source": "state_pool_numeric_channels",
            "playback_policy": playback_allowed,
        },
        "audio_sources": [],
        "feature_summary": feature_summary,
        "reconstruction_diagnostics": diagnostics,
        "audio_mix": mix_trace,
        "focus_payload_channels": sorted(dict((audio.get("focus", {}) or {}).get("payloads", {}) or {})),
        "event_sources": [
            {
                "event_id": str(row.get("event_id", "") or ""),
                "layer_type": _render_layer_type(str(row.get("source", "real_input") or "real_input")),
                "opacity": float(row.get("opacity", 0.0) or 0.0),
                "channels": sorted(dict((row.get("reconstruction_payload", {}) or {}).get("channels", {}) or {})),
                "sampling_focus": dict(row.get("sampling_focus", {}) or {}),
                "not_new_external_input": bool(row.get("not_new_external_input", False)),
                "echo_kind": str(row.get("echo_kind", "") or ""),
                "echo_modality": str(row.get("echo_modality", "") or ""),
                "age_ticks": int(row.get("age_ticks", 0) or 0),
            }
            for row in events[:8]
        ],
        "predicted_labels": list(audio.get("predicted_labels", []) or []),
    }


def _vision_background(objects: list[dict]) -> dict:
    if not objects:
        return {
            "mode": "state_pool_neutral_wash",
            "mean_rgb": [0.93, 0.95, 0.97],
            "reconstruction_basis": "state_pool_numeric_channels",
        }
    total = 0.0
    accum = [0.0, 0.0, 0.0]
    for obj in objects:
        rgb = list(obj.get("mean_rgb", []) or [])
        if len(rgb) < 3:
            continue
        weight = max(0.05, float(obj.get("opacity", 0.0) or 0.0))
        total += weight
        for idx in range(3):
            accum[idx] += max(0.0, min(1.0, float(rgb[idx] or 0.0))) * weight
    if total <= 1e-9:
        return {
            "mode": "state_pool_neutral_wash",
            "mean_rgb": [0.93, 0.95, 0.97],
            "reconstruction_basis": "state_pool_numeric_channels",
        }
    return {
        "mode": "state_pool_color_wash",
        "mean_rgb": [round(value / total, 4) for value in accum],
        "reconstruction_basis": "state_pool_numeric_channels",
    }


def _build_detail_panel(observatory_payload: dict, *, vision: dict, audio: dict) -> dict:
    payload = dict(observatory_payload or {})
    fast = dict(payload.get("fast_system", {}) or {})
    slow = dict(payload.get("slow_system", {}) or {})
    state_pool = dict(payload.get("state_pool", {}) or {})
    focus = dict((vision or {}).get("focus_overlay", {}) or {})
    return {
        "schema_id": "apv21_observatory_detail_panel/v1",
        "tick_index": payload.get("tick_index"),
        "memory_layers": {
            "fast_bn": [_bn_detail_row(row, layer_id="fast_bn") for row in list(fast.get("bn", []) or [])[:4]],
            "slow_bn_prime": [_bn_detail_row(row, layer_id="slow_bn_prime") for row in list(slow.get("bn_prime", []) or [])[:4]],
            "fast_cn": [_cn_detail_row(row, layer_id="fast_cn") for row in list(fast.get("cn", []) or [])[:4]],
            "slow_cn_prime": [_cn_detail_row(row, layer_id="slow_cn_prime") for row in list(slow.get("cn_prime", []) or [])[:4]],
        },
        "short_term_echo": _short_term_echo_detail(payload),
        "action": dict(payload.get("action", {}) or {}),
        "prediction_summary": _prediction_summary(fast=fast, slow=slow, vision=vision, audio=audio),
        "focus_summary": {
            "schema_id": "apv21_focus_detail_summary/v1",
            "center_norm": list(focus.get("center_norm", []) or []),
            "radius_norm": float(focus.get("radius_norm", 0.0) or 0.0),
            "clarity": dict(focus.get("clarity", {}) or {}),
            "resolution_summary": dict(focus.get("resolution_summary", {}) or {}),
            "object_focus": list(focus.get("object_focus", []) or [])[:8],
            "audio_focus_band": dict((audio or {}).get("focus_band_overlay", {}) or {}),
        },
        "top_state_items": list(state_pool.get("top_items", []) or [])[:8],
    }


def _short_term_echo_detail(observatory_payload: dict) -> dict:
    echo = dict(((observatory_payload or {}).get("inner_world", {}) or {}).get("short_term_echo", {}) or {})
    if not echo:
        echo = {"schema_id": "apv21_short_term_echo_observatory/v1", "applied": False, "echo_count": 0, "source_counts": {}, "items_preview": []}
    return {
        "schema_id": str(echo.get("schema_id", "") or "apv21_short_term_echo_observatory/v1"),
        "applied": bool(echo.get("applied", False)),
        "echo_count": int(echo.get("echo_count", 0) or 0),
        "source_counts": dict(echo.get("source_counts", {}) or {}),
        "items_preview": list(echo.get("items_preview", []) or [])[:8],
        "policy": str(echo.get("policy", "") or ""),
    }


def _bn_detail_row(row: dict, *, layer_id: str) -> dict:
    items = list((row or {}).get("core_items", []) or (row or {}).get("items", []) or [])
    energy = dict((row or {}).get("energy_summary", {}) or {})
    return {
        "layer_id": layer_id,
        "memory_id": str((row or {}).get("memory_id", "") or ""),
        "memory_kind": str((row or {}).get("memory_kind", "") or ""),
        "tick_index": int((row or {}).get("tick_index", -1) or -1),
        "source_text": str((row or {}).get("source_text", "") or ""),
        "score": float((row or {}).get("bn_score", (row or {}).get("score", 0.0)) or 0.0),
        "normalized_weight": float((row or {}).get("normalized_weight", 0.0) or 0.0),
        "energy_total": float(energy.get("total_real_energy", 0.0) or 0.0) + float(energy.get("total_virtual_energy", 0.0) or 0.0),
        "top_labels": [str((item or {}).get("sa_label", "") or "") for item in items[:6]],
        "score_breakdown": dict((row or {}).get("score_breakdown", {}) or {}),
        "candidate_sources": list((row or {}).get("candidate_sources", []) or [])[:8],
    }


def _cn_detail_row(row: dict, *, layer_id: str) -> dict:
    predicted_labels = list((row or {}).get("predicted_labels", []) or [])
    successor = dict((row or {}).get("successor", {}) or {})
    successor_items = list(successor.get("prediction_payload_items", []) or successor.get("core_items", []) or successor.get("items", []) or [])
    virtual_sum = 0.0
    for item in successor_items:
        if isinstance(item, dict):
            virtual_sum += float(item.get("virtual_energy", item.get("real_energy", 0.0)) or 0.0)
    return {
        "layer_id": layer_id,
        "source_memory_id": str((row or {}).get("source_memory_id", "") or ""),
        "successor_memory_id": str((row or {}).get("successor_memory_id", "") or ""),
        "memory_id": str((row or {}).get("successor_memory_id", "") or ""),
        "score": float((row or {}).get("score", 0.0) or 0.0),
        "learned_transition_score": float((row or {}).get("learned_transition_score", 0.0) or 0.0),
        "predicted_labels": predicted_labels[:8],
        "top_labels": predicted_labels[:6],
        "virtual_energy_sum": round(virtual_sum, 4),
        "successor_tick_index": int(successor.get("tick_index", -1) or -1),
    }


def _prediction_summary(*, fast: dict, slow: dict, vision: dict, audio: dict) -> dict:
    labels: dict[str, float] = {}
    for branch in list(fast.get("cn", []) or []) + list(slow.get("cn_prime", []) or []):
        for label in list((branch or {}).get("predicted_labels", []) or []):
            clean = str(label or "")
            if clean:
                labels[clean] = labels.get(clean, 0.0) + 1.0
    rows = [
        {"sa_label": label, "support": round(value, 4)}
        for label, value in sorted(labels.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]
    return {
        "schema_id": "apv21_prediction_detail_summary/v1",
        "top_predicted_labels": rows,
        "vision_predicted_labels": list((vision or {}).get("predicted_labels", []) or [])[:12],
        "audio_predicted_labels": list((audio or {}).get("predicted_labels", []) or [])[:12],
    }


def _render_layer_type(source: str) -> str:
    clean = str(source or "").strip()
    if clean in {"real_input", "real"}:
        return "real"
    if clean in {"recalled_memory", "memory_completion", "recalled"}:
        return "recalled"
    if clean in {"short_term_echo", "sensory_echo", "thought_echo", "echo"}:
        return "echo"
    if clean in {"cstar_prediction", "predicted", "prediction"}:
        return "predicted"
    return clean.replace("_input", "") or "real"


def _synthetic_spectrum(*, band_count: int, centroid: float, rms: float) -> list[float]:
    count = max(1, int(band_count))
    center = min(count - 1, max(0, int((max(0.0, centroid) / 8000.0) * count)))
    base = max(0.04, min(1.0, rms * 6.0))
    rows = []
    for idx in range(count):
        distance = abs(idx - center)
        rows.append(round(max(0.025, base * math.exp(-distance / max(1.0, count / 5.0))), 4))
    return rows


def _synthetic_waveform(*, rms: float, dominant_hz: float, onset: float, points: int = 96) -> list[float]:
    amp = max(0.04, min(0.95, float(rms or 0.0) * 5.0 + float(onset or 0.0) * 8.0))
    cycles = max(1.0, min(9.0, float(dominant_hz or 220.0) / 110.0))
    return [round(math.sin((idx / max(1, points - 1)) * math.tau * cycles) * amp, 4) for idx in range(points)]


def _payload_values(payload: dict) -> list[float]:
    if not isinstance(payload, dict):
        return []
    values = payload.get("payload_values", [])
    if not isinstance(values, (list, tuple)):
        return []
    rows = []
    for value in values:
        try:
            rows.append(float(value))
        except (TypeError, ValueError):
            rows.append(0.0)
    return rows


def _event_payload(event: dict, key: str, channel: str = "") -> dict:
    payload = event.get(key, {}) if isinstance(event, dict) else {}
    if isinstance(payload, dict) and payload:
        return payload
    bundle = dict((event or {}).get("reconstruction_payload", {}) or {}) if isinstance(event, dict) else {}
    channels = dict(bundle.get("channels", {}) or {})
    return dict(channels.get(channel, {}) or {}) if channel else {}


def _audio_reconstruction_diagnostics(*, event: dict, preview: dict, playback_source: dict, duration_ms: float, mix_trace: dict | None = None) -> dict:
    waveform = _event_payload(event, "waveform_payload", "audio.focus.waveform_slice")
    waveform_values = _payload_values(waveform)
    feature_summary = dict((event or {}).get("feature_summary", {}) or {})
    speech_diag = dict(feature_summary.get("speech_like_reconstruction", {}) or {})
    input_duration = float(preview.get("preview_duration_ms", 0.0) or 0.0)
    coverage = float(speech_diag.get("waveform_coverage_ratio", 0.0) or 0.0)
    if not speech_diag and input_duration > 0.0 and waveform_values:
        sample_rate = max(1.0, float(preview.get("sample_rate", 16000) or 16000))
        coverage = min(1.0, len(waveform_values) / max(1.0, input_duration * sample_rate / 1000.0))
    if coverage >= 0.78 and duration_ms >= min(input_duration or duration_ms, 1450.0) * 0.85:
        risk = "low_numeric_slice_preserves_most_short_phrase_waveform"
    elif coverage >= 0.35:
        risk = "medium_missing_some_phoneme_detail"
    else:
        risk = "high_payload_is_acoustic_gist_not_clear_speech"
    mix = dict(mix_trace or {})
    return {
        "schema_id": "audio_reconstruction_diagnostics/v1",
        "input_duration_ms": round(input_duration, 3),
        "playback_duration_ms": round(float(duration_ms), 3),
        "waveform_payload_points": int(len(waveform_values)),
        "waveform_coverage_ratio": round(max(0.0, min(1.0, coverage)), 4),
        "numeric_channels_used": list((playback_source or {}).get("numeric_channels_used", []) or []),
        "playback_suppressed": not bool(playback_source),
        "asset_shortcut_used": False,
        "energy_weighted_loudness": bool(mix.get("energy_weighted_loudness", False)),
        "mixed_event_count": int(mix.get("mixed_event_count", 0) or 0),
        "dominant_event_id": str(mix.get("dominant_event_id", "") or ""),
        "dominant_mix_event_key": str(mix.get("dominant_mix_event_key", "") or ""),
        "event_mix_weights": list(mix.get("event_mix_weights", []) or [])[:8],
        "intelligibility_risk": risk,
        "meaning": "inner_audio_is_reconstructed_from_state_pool_numeric_payloads_not_raw_demo_audio",
    }


def _audio_playback_allowed(events: list[dict], *, preview: dict) -> dict:
    """
    Gate the browser audio player on interpretable numeric waveform evidence.

    Spectrum/envelope/pitch payloads are useful for the inner-audio panel, but
    they are not enough to replay understandable speech. Rendering them as a WAV
    creates a misleading noise-like artifact, so the player stays hidden unless
    AP's state pool carries a real waveform slice with meaningful coverage.
    """

    waveform_points = 0
    best_coverage = 0.0
    input_duration = float((preview or {}).get("preview_duration_ms", 0.0) or 0.0)
    sample_rate = max(1.0, float((preview or {}).get("sample_rate", 16000) or 16000))
    for event in events or []:
        if not isinstance(event, dict):
            continue
        waveform = _payload_values(_event_payload(event, "waveform_payload", "audio.focus.waveform_slice"))
        if not waveform:
            continue
        waveform_points = max(waveform_points, len(waveform))
        speech_diag = dict((dict(event.get("feature_summary", {}) or {}).get("speech_like_reconstruction", {}) or {}))
        coverage = float(speech_diag.get("waveform_coverage_ratio", 0.0) or 0.0)
        if coverage <= 0.0 and input_duration > 0.0:
            coverage = min(1.0, len(waveform) / max(1.0, input_duration * sample_rate / 1000.0))
        best_coverage = max(best_coverage, coverage)
    allowed = waveform_points >= 64 and (best_coverage >= 0.22 or input_duration <= 0.0)
    reason = "waveform_payload_sufficient_for_playback" if allowed else "insufficient_waveform_payload_for_understandable_audio"
    return {
        "schema_id": "audio_playback_gate/v1",
        "allowed": bool(allowed),
        "reason": reason,
        "waveform_payload_points": int(waveform_points),
        "best_waveform_coverage_ratio": round(max(0.0, min(1.0, best_coverage)), 4),
        "spectrum_or_envelope_still_displayed": True,
    }


def _resample_list(values: list[float], points: int) -> list[float]:
    count = max(1, int(points))
    if not values:
        return []
    if len(values) == count:
        return [round(float(value), 4) for value in values]
    if len(values) == 1:
        return [round(float(values[0]), 4)] * count
    rows = []
    for idx in range(count):
        pos = idx / max(1, count - 1) * (len(values) - 1)
        left = int(math.floor(pos))
        right = min(len(values) - 1, left + 1)
        frac = pos - left
        rows.append(round(float(values[left]) * (1.0 - frac) + float(values[right]) * frac, 4))
    return rows


def _audio_event_mix_trace(events: list[dict], *, limit: int = 6) -> dict:
    candidates = []
    for event_index, event in enumerate(events or []):
        if not isinstance(event, dict):
            continue
        if not _event_has_audio_payload(event):
            continue
        source = str(event.get("source", "") or "audio_event")
        raw = _audio_event_loudness_energy(event)
        if raw <= 0.0:
            continue
        # Current input is usually clearer because it came from this tick's
        # focused sensor payload. Recalled/predicted events still participate,
        # but their lower energy keeps them as background imagination unless
        # their B/C support is genuinely strong.
        if source in {"real_input", "real"}:
            raw *= 1.0
        elif source in {"recalled_memory", "memory_completion"}:
            raw *= 0.72
        elif source in {"short_term_echo", "sensory_echo", "thought_echo", "echo"}:
            raw *= 0.54
        elif source in {"cstar_prediction", "predicted", "prediction"}:
            raw *= 0.62
        candidates.append((event_index, dict(event), max(0.0, raw)))
    candidates.sort(
        key=lambda row: (
            -float(row[2]),
            str(row[1].get("source", "") or ""),
            str(row[1].get("event_id", "") or ""),
        )
    )
    selected = candidates[: max(1, int(limit))]
    total = sum(weight for _, _, weight in selected)
    if total <= 1e-9:
        return {
            "schema_id": "audio_energy_weighted_mix/v1",
            "energy_weighted_loudness": False,
            "mixed_event_count": 0,
            "dominant_event_id": "",
            "event_mix_weights": [],
            "policy": "no_numeric_audio_payload_to_mix",
        }
    rows = []
    for event_index, event, weight in selected:
        norm = weight / total
        event_id = str(event.get("event_id", "") or event.get("source", "") or "audio_event")
        rows.append(
            {
                "mix_event_key": _audio_event_mix_key(event, event_index),
                "event_index": int(event_index),
                "event_id": event_id,
                "source": str(event.get("source", "") or ""),
                "raw_energy": round(float(weight), 4),
                "weight": round(float(norm), 4),
                "opacity": round(float(event.get("opacity", 0.0) or 0.0), 4),
                "virtual_energy": round(float(event.get("virtual_energy", 0.0) or 0.0), 4),
            }
        )
    return {
        "schema_id": "audio_energy_weighted_mix/v1",
        "energy_weighted_loudness": True,
        "mixed_event_count": len(rows),
        "dominant_event_id": rows[0]["event_id"] if rows else "",
        "dominant_mix_event_key": rows[0]["mix_event_key"] if rows else "",
        "event_mix_weights": rows,
        "policy": "state_pool_numeric_audio_events_mixed_by_energy_without_asset_replay",
    }


def _audio_event_mix_key(event: dict, event_index: int) -> str:
    source = str((event or {}).get("source", "") or "audio_event")
    event_id = str((event or {}).get("event_id", "") or source)
    return f"{int(event_index)}::{source}::{event_id}"


def _event_has_audio_payload(event: dict) -> bool:
    for key, channel in (
        ("waveform_payload", "audio.focus.waveform_slice"),
        ("envelope_payload", "audio.focus.envelope"),
        ("pitch_contour_payload", "audio.focus.pitch_contour"),
        ("stft_magnitude_payload", "audio.focus.stft_magnitude"),
        ("stft_phase_payload", "audio.focus.stft_phase"),
    ):
        if _payload_values(_event_payload(event, key, channel)):
            return True
    return False


def _audio_event_loudness_energy(event: dict) -> float:
    opacity = max(0.0, float((event or {}).get("opacity", 0.0) or 0.0))
    salience = max(0.0, float((event or {}).get("salience", 0.0) or 0.0))
    virtual = max(0.0, float((event or {}).get("virtual_energy", 0.0) or 0.0))
    features = dict((event or {}).get("feature_summary", {}) or {})
    rms = max(0.0, float(features.get("rms", 0.0) or 0.0))
    onset = max(0.0, float(features.get("onset_strength", 0.0) or 0.0))
    payload = _event_payload(event, "waveform_payload", "audio.focus.waveform_slice")
    payload_precision = max(0.0, min(1.0, float(payload.get("sampling_precision", 0.0) or 0.0))) if isinstance(payload, dict) else 0.0
    return max(0.0, opacity * 0.70 + salience * 0.26 + virtual * 0.38 + rms * 1.8 + onset * 1.2 + payload_precision * 0.16)


def _waveform_preview_from_audio_payload(event: dict, *, points: int) -> list[float]:
    waveform = _payload_values(_event_payload(event, "waveform_payload", "audio.focus.waveform_slice"))
    if waveform:
        peak = max(1e-9, max(abs(value) for value in waveform))
        return [round(max(-1.0, min(1.0, value / peak)), 4) for value in _resample_list(waveform, points)]
    envelope = _payload_values(_event_payload(event, "envelope_payload", "audio.focus.envelope"))
    pitch = _payload_values(_event_payload(event, "pitch_contour_payload", "audio.focus.pitch_contour"))
    if not envelope:
        return []
    env = _resample_list(envelope, points)
    pitch_curve = _resample_list(pitch, points) if pitch else [0.06] * points
    phase = 0.0
    rows = []
    for idx, amp in enumerate(env):
        freq_norm = max(0.01, min(0.45, abs(float(pitch_curve[idx] if idx < len(pitch_curve) else 0.06))))
        phase += math.tau * (1.0 + freq_norm * 16.0) / max(1, points)
        rows.append(round(math.sin(phase) * max(0.0, min(1.0, float(amp))), 4))
    return rows


def _spectrum_from_audio_payload(event: dict, *, band_count: int) -> list[float]:
    stft = _event_payload(event, "stft_magnitude_payload", "audio.focus.stft_magnitude")
    values = _payload_values(stft)
    shape = list(stft.get("payload_shape", []) or []) if isinstance(stft, dict) else []
    if len(shape) >= 2 and values:
        rows = max(1, int(shape[0] or 1))
        cols = max(1, int(shape[1] or 1))
        matrix = values[: rows * cols]
        bands = []
        for col in range(cols):
            total = 0.0
            count = 0
            for row in range(rows):
                idx = row * cols + col
                if idx < len(matrix):
                    total += max(0.0, float(matrix[idx]))
                    count += 1
            bands.append(total / max(1, count))
        peak = max(1e-9, max(bands) if bands else 0.0)
        return [round(max(0.02, value / peak), 4) for value in _resample_list(bands, band_count)]
    bundle = dict((event or {}).get("reconstruction_payload", {}) or {})
    channels = dict(bundle.get("channels", {}) or {})
    band_payload = channels.get("audio.global.band") if isinstance(channels, dict) else None
    band_values = _payload_values(band_payload) if isinstance(band_payload, dict) else []
    return _resample_list(band_values, band_count) if band_values else []


def _audio_band_focus_gain(*, idx: int, band_count: int, focus_band_overlay: dict) -> float:
    if not focus_band_overlay:
        return 0.0
    count = max(1, int(band_count))
    center = float(focus_band_overlay.get("center_norm", 0.0) or 0.0)
    low = float(focus_band_overlay.get("low_norm", 0.0) or 0.0)
    high = float(focus_band_overlay.get("high_norm", 1.0) or 1.0)
    pos = (float(idx) + 0.5) / float(count)
    if pos < low or pos > high:
        edge = min(abs(pos - low), abs(pos - high))
        return round(max(0.0, 0.22 - edge * 1.4), 4)
    half_width = max(0.02, (high - low) * 0.5)
    return round(max(0.24, min(1.0, 1.0 - abs(pos - center) / max(half_width, 0.02) * 0.55)), 4)


def _synthesized_audio_source(
    event: dict,
    waveform_preview: list[float],
    *,
    sample_rate: int,
    duration_ms: float,
    events: list[dict] | None = None,
    mix_trace: dict | None = None,
) -> dict:
    if not _numeric_audio_channels_used(events or [event]):
        return {}
    clean_sample_rate = max(8000, int(sample_rate))
    clean_duration = max(80.0, float(duration_ms))
    samples = _mixed_samples_from_audio_events(
        events or [event],
        fallback_event=event,
        fallback_preview=waveform_preview,
        sample_rate=clean_sample_rate,
        duration_ms=clean_duration,
        mix_trace=mix_trace or {},
    )
    if not samples:
        return {}
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(clean_sample_rate)
        frames = [
            struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767.0))
            for sample in samples
        ]
        wav_file.writeframes(b"".join(frames))
    return {
        "role": "state_pool_synthesis",
        "layer_type": "real",
        "encoding": "wav",
        "playable": True,
        "asset_id": "",
        "playback_url": "",
        "data_url": f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}",
        "reconstruction_basis": "state_pool_numeric_channels",
        "numeric_channels_used": _numeric_audio_channels_used(events or [event]),
        "energy_weighted_loudness": bool((mix_trace or {}).get("energy_weighted_loudness", False)),
        "mixed_event_count": int((mix_trace or {}).get("mixed_event_count", 0) or 0),
    }


def _numeric_audio_channels_used(events: list[dict]) -> list[str]:
    channels = []
    seen = set()
    for event in events or []:
        for channel, payload in (
            ("audio.focus.waveform_slice", _event_payload(event, "waveform_payload", "audio.focus.waveform_slice")),
            ("audio.focus.envelope", _event_payload(event, "envelope_payload", "audio.focus.envelope")),
            ("audio.focus.pitch_contour", _event_payload(event, "pitch_contour_payload", "audio.focus.pitch_contour")),
            ("audio.focus.stft_magnitude", _event_payload(event, "stft_magnitude_payload", "audio.focus.stft_magnitude")),
            ("audio.focus.stft_phase", _event_payload(event, "stft_phase_payload", "audio.focus.stft_phase")),
            ("audio.focus.onset_events", _event_payload(event, "onset_events_payload", "audio.focus.onset_events")),
            ("audio.focus.transient", _event_payload(event, "transient_payload", "audio.focus.transient")),
            ("audio.focus.harmonic_noise", _event_payload(event, "harmonic_noise_payload", "audio.focus.harmonic_noise")),
        ):
            if channel in seen:
                continue
            if isinstance(payload, dict) and payload.get("payload_values"):
                seen.add(channel)
                channels.append(channel)
    return channels


def _mixed_samples_from_audio_events(
    events: list[dict],
    *,
    fallback_event: dict,
    fallback_preview: list[float],
    sample_rate: int,
    duration_ms: float,
    mix_trace: dict,
) -> list[float]:
    sample_count = max(1, int(max(8000, sample_rate) * max(80.0, duration_ms) / 1000.0))
    mix_rows = list((mix_trace or {}).get("event_mix_weights", []) or [])
    event_by_key = {
        _audio_event_mix_key(event, idx): dict(event)
        for idx, event in enumerate(events or [])
        if isinstance(event, dict)
    }
    accum = [0.0] * sample_count
    used = 0
    for row in mix_rows:
        event_key = str((row or {}).get("mix_event_key", "") or "")
        event = event_by_key.get(event_key)
        if not event:
            try:
                event_index = int((row or {}).get("event_index", -1))
            except (TypeError, ValueError):
                event_index = -1
            if 0 <= event_index < len(events or []) and isinstance((events or [])[event_index], dict):
                event = dict((events or [])[event_index])
        if not event:
            continue
        weight = max(0.0, min(1.0, float((row or {}).get("weight", 0.0) or 0.0)))
        if weight <= 0.0:
            continue
        samples = _synthesize_samples_from_state_features(
            dict(event.get("feature_summary", {}) or {}),
            _waveform_preview_from_audio_payload(event, points=128),
            waveform_payload=_event_payload(event, "waveform_payload", "audio.focus.waveform_slice"),
            envelope_payload=_event_payload(event, "envelope_payload", "audio.focus.envelope"),
            pitch_payload=_event_payload(event, "pitch_contour_payload", "audio.focus.pitch_contour"),
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )
        if not samples:
            continue
        if len(samples) != sample_count:
            samples = _resample_list(samples, sample_count)
        for idx, sample in enumerate(samples[:sample_count]):
            accum[idx] += float(sample) * weight
        used += 1
    if used <= 0:
        return _synthesize_samples_from_state_features(
            dict((fallback_event or {}).get("feature_summary", {}) or {}),
            fallback_preview,
            waveform_payload=_event_payload(fallback_event, "waveform_payload", "audio.focus.waveform_slice"),
            envelope_payload=_event_payload(fallback_event, "envelope_payload", "audio.focus.envelope"),
            pitch_payload=_event_payload(fallback_event, "pitch_contour_payload", "audio.focus.pitch_contour"),
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )
    return _postprocess_inner_audio_samples(accum)


def _postprocess_inner_audio_samples(samples: list[float]) -> list[float]:
    if not samples:
        return []
    # A tiny smoothing pass removes zipper noise from resampled numeric payloads
    # while preserving the waveform slice's speech-like shape. This is still a
    # render-only reconstruction step and never feeds back into AP cognition.
    smoothed = []
    previous = float(samples[0])
    for sample in samples:
        current = float(sample)
        value = previous * 0.18 + current * 0.82
        smoothed.append(value)
        previous = value
    peak = max(1e-9, max(abs(value) for value in smoothed))
    gain = min(0.92, 0.78 / peak) if peak > 0.78 else 1.0
    return [max(-1.0, min(1.0, value * gain)) for value in smoothed]


def _synthesize_samples_from_state_features(
    feature_summary: dict,
    waveform_preview: list[float],
    *,
    waveform_payload: dict | None = None,
    envelope_payload: dict | None = None,
    pitch_payload: dict | None = None,
    sample_rate: int,
    duration_ms: float,
) -> list[float]:
    sample_count = max(1, int(max(8000, sample_rate) * max(80.0, duration_ms) / 1000.0))
    waveform_values = _payload_values(waveform_payload or {})
    if waveform_values:
        rows = _resample_list(waveform_values, sample_count)
        peak = max(1e-9, max(abs(value) for value in rows))
        return [max(-1.0, min(1.0, float(value) / peak * 0.86)) for value in rows]
    dominant = max(40.0, min(4000.0, float(feature_summary.get("dominant_hz", 0.0) or 220.0)))
    centroid = max(40.0, min(8000.0, float(feature_summary.get("spectral_centroid_hz", 0.0) or dominant)))
    rms = max(0.0, float(feature_summary.get("rms", 0.0) or 0.0))
    onset = max(0.0, float(feature_summary.get("onset_strength", 0.0) or 0.0))
    amp = max(0.035, min(0.72, rms * 4.0 + onset * 6.0))
    harmonic = max(dominant, min(8000.0, centroid))
    rows = []
    preview = [float(value or 0.0) for value in waveform_preview] or [0.0]
    envelope_curve = _resample_list(_payload_values(envelope_payload or {}), sample_count)
    pitch_curve = _resample_list(_payload_values(pitch_payload or {}), sample_count)
    phase = 0.0
    for idx in range(sample_count):
        t = idx / float(max(1, sample_rate))
        p = idx / float(max(1, sample_count - 1))
        env_idx = min(len(preview) - 1, int(p * max(0, len(preview) - 1)))
        preview_env = 0.55 + 0.45 * min(1.0, abs(preview[env_idx]))
        attack = min(1.0, idx / max(1.0, sample_rate * 0.018))
        release = min(1.0, (sample_count - idx) / max(1.0, sample_rate * 0.035))
        envelope = max(0.0, min(1.0, attack, release)) * preview_env
        if envelope_curve:
            envelope *= max(0.0, min(1.0, float(envelope_curve[idx])))
        if pitch_curve:
            dominant = max(40.0, min(4000.0, float(pitch_curve[idx]) * sample_rate * 0.5))
            harmonic = max(dominant, min(8000.0, dominant * 1.7))
        phase += math.tau * dominant / max(1, sample_rate)
        value = math.sin(phase)
        value += 0.28 * math.sin(math.tau * min(8000.0, harmonic) * t)
        value += 0.12 * math.sin(math.tau * min(8000.0, dominant * 2.0) * t)
        rows.append(max(-1.0, min(1.0, value * amp * envelope / 1.4)))
    return rows


def _layer_stack(vision: dict, audio: dict) -> dict:
    return {
        "schema_id": "apv21_realtime_overlay_layer_stack/v1",
        "vision": [
            _layer_stack_row("current_input", _has_layer(vision, "real"), "current visual input reconstructed from numeric state-pool payloads"),
            _layer_stack_row("recalled_memory", _has_layer(vision, "recalled"), "Bn/Bn' visual memory completion layer"),
            _layer_stack_row(
                "short_term_echo",
                _has_layer(vision, "short_term_echo") or _has_layer(vision, "echo"),
                "decayed visual afterimage/recent residue marked as not current external input",
            ),
            _layer_stack_row(
                "cstar_prediction",
                _has_layer(vision, "predicted") or bool((vision or {}).get("predicted_labels")),
                "Cn/Cn'/C* visual prediction layer carrying virtual-energy predictions",
            ),
            _layer_stack_row(
                "focus_precision",
                bool(((vision or {}).get("focus_overlay", {}) or {}).get("visible", False)),
                "foveated sampling precision derived from gaze state and sensor precision grid",
            ),
        ],
        "audio": [
            _layer_stack_row("current_input", _has_layer(audio, "real"), "current audio input reconstructed from numeric state-pool payloads"),
            _layer_stack_row("recalled_memory", _has_layer(audio, "recalled"), "Bn/Bn' audio memory completion layer"),
            _layer_stack_row(
                "short_term_echo",
                _has_layer(audio, "short_term_echo") or _has_layer(audio, "echo"),
                "decayed aftersound/recent auditory residue marked as not current external input",
            ),
            _layer_stack_row(
                "cstar_prediction",
                _has_layer(audio, "predicted") or bool((audio or {}).get("predicted_labels")),
                "Cn/Cn'/C* audio prediction layer carrying virtual-energy predictions",
            ),
            _layer_stack_row(
                "focus_band",
                bool(((audio or {}).get("focus_band_overlay", {}) or {}).get("visible", False))
                or bool((((audio or {}).get("focus", {}) or {}).get("focus_band_overlay", {}) or {}).get("visible", False)),
                "focused listening band derived from auditory-band actuator state",
            ),
        ],
    }


def _layer_stack_row(role: str, visible: bool, description: str) -> dict:
    return {
        "role": role,
        "visible": bool(visible),
        "description": description,
    }


def _has_layer(payload: dict, layer_type: str) -> bool:
    return any(
        isinstance(row, dict) and str(row.get("layer_type", "") or "") == layer_type
        for row in list((payload or {}).get("layers", []) or [])
    )


def _layer_legend(vision: dict, audio: dict) -> list[dict]:
    rows = []
    for source, payload in (("vision", vision), ("audio", audio)):
        for layer in list((payload or {}).get("layers", []) or []):
            if not isinstance(layer, dict):
                continue
            rows.append(
                {
                    "source": source,
                    "layer_id": str(layer.get("layer_id", "") or ""),
                    "layer_type": str(layer.get("layer_type", "") or ""),
                    "label": str(layer.get("description", "") or layer.get("layer_id", "") or ""),
                }
            )
    return rows
