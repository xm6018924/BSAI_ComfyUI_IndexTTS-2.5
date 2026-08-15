/**
 * BSAI Audio Upload Extension
 *
 * Adds "加载音频" (Load Audio) button and an audio player bar with
 * progress bar and time display to BSAI_IndexTTS2.5LoadAudio nodes.
 */

const app = window.comfyAPI?.app?.app ?? window.app;
const NODE_TYPE = "BSAI_IndexTTS2.5LoadAudio";

app.registerExtension({
    name: "BSAI.IndexTTS2.5.AudioUpload",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const audioWidget = this.widgets?.find((w) => w.name === "audio_音频" || w.name === "audio");
            if (!audioWidget) {
                setTimeout(() => setupUpload(this), 0);
                return;
            }
            setupUpload(this);
        };
    },
});

function formatTime(sec) {
    if (!sec || isNaN(sec)) return "0:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
}

function setupUpload(node) {
    const audioWidget = node.widgets?.find((w) => w.name === "audio_音频" || w.name === "audio");
    if (!audioWidget) return;
    if (node._bsaiUploadSetup) return;
    node._bsaiUploadSetup = true;

    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width: 100%; padding: 2px 0;";

    // --- 加载音频 button ---
    const loadBtn = document.createElement("button");
    loadBtn.textContent = "📁 加载音频";
    loadBtn.style.cssText = `
        display: block; width: 100%; padding: 5px 8px; margin-bottom: 4px;
        background: #2a2a3e; color: #88aacc; border: 1px solid #3a5a7a;
        border-radius: 4px; cursor: pointer; font-size: 12px; text-align: center;
        transition: background 0.15s;
    `;
    loadBtn.onmouseenter = () => { loadBtn.style.background = "#3a3a4e"; };
    loadBtn.onmouseleave = () => { loadBtn.style.background = "#2a2a3e"; };

    // --- Hidden file input ---
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".wav,.mp3,.flac,.ogg,.m4a";
    fileInput.style.display = "none";

    loadBtn.onclick = () => fileInput.click();

    fileInput.onchange = async () => {
        const file = fileInput.files[0];
        if (!file) return;

        loadBtn.textContent = "⏳ 加载中...";
        loadBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append("image", file);
            formData.append("type", "input");
            formData.append("overwrite", "true");

            const resp = await fetch("/upload/image", {
                method: "POST",
                body: formData,
            });

            if (!resp.ok) throw new Error(`Upload failed: ${resp.statusText}`);

            const data = await resp.json();
            audioWidget.value = data.name;

            if (audioWidget.options && Array.isArray(audioWidget.options.values)) {
                if (!audioWidget.options.values.includes(data.name)) {
                    audioWidget.options.values.push(data.name);
                    audioWidget.options.values.sort();
                }
            }

            updateAudioPreview(data.name);
        } catch (err) {
            console.error("[BSAI] Upload error:", err);
            alert(`加载失败: ${err.message}`);
        } finally {
            loadBtn.textContent = "📁 加载音频";
            loadBtn.disabled = false;
            fileInput.value = "";
        }
    };

    // --- Audio player bar ---
    const playerBar = document.createElement("div");
    playerBar.style.cssText = `
        display: none; align-items: center; gap: 6px; width: 100%;
        padding: 4px 6px; background: #1e1e2e; border: 1px solid #333344;
        border-radius: 4px;
    `;

    // Play/pause button
    const playBtn = document.createElement("button");
    playBtn.textContent = "▶";
    playBtn.style.cssText = `
        flex-shrink: 0; width: 26px; height: 26px; padding: 0;
        background: #2a4a2a; color: #aacc88; border: 1px solid #5a7a3a;
        border-radius: 4px; cursor: pointer; font-size: 12px;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.15s;
    `;
    playBtn.onmouseenter = () => { playBtn.style.background = "#3a5a3a"; };
    playBtn.onmouseleave = () => { playBtn.style.background = "#2a4a2a"; };

    // Time display
    const timeDisplay = document.createElement("span");
    timeDisplay.textContent = "0:00 / 0:00";
    timeDisplay.style.cssText = `
        flex-shrink: 0; color: #aaaacc; font-size: 11px;
        font-family: monospace; min-width: 64px; text-align: center;
    `;

    // Progress bar
    const progressContainer = document.createElement("div");
    progressContainer.style.cssText = `
        flex: 1; height: 6px; background: #333344; border-radius: 3px;
        cursor: pointer; position: relative; overflow: hidden;
    `;

    const progressFill = document.createElement("div");
    progressFill.style.cssText = `
        width: 0%; height: 100%; background: #5a7a9a; border-radius: 3px;
        transition: width 0.1s linear;
    `;
    progressContainer.appendChild(progressFill);

    // Volume icon
    const volIcon = document.createElement("span");
    volIcon.textContent = "🔊";
    volIcon.style.cssText = `
        flex-shrink: 0; font-size: 12px; cursor: pointer; opacity: 0.7;
    `;
    volIcon.title = "点击切换静音";
    let muted = false;
    volIcon.onclick = () => {
        muted = !muted;
        audioEl.muted = muted;
        volIcon.textContent = muted ? "🔇" : "🔊";
    };

    // --- Audio element (hidden) ---
    const audioEl = document.createElement("audio");
    audioEl.preload = "metadata";
    audioEl.style.cssText = "display: none;";

    // --- Update functions ---
    function updateAudioPreview(filename) {
        if (filename && !filename.startsWith("upload_")) {
            audioEl.src = `/api/view?filename=${encodeURIComponent(filename)}&type=input`;
            playerBar.style.display = "flex";
        } else {
            audioEl.src = "";
            audioEl.pause();
            playerBar.style.display = "none";
        }
    }

    // --- Audio event handlers ---
    playBtn.onclick = () => {
        if (!audioEl.src) return;
        if (audioEl.paused) {
            audioEl.play().catch(err => console.error("[BSAI] Playback error:", err));
        } else {
            audioEl.pause();
        }
    };

    audioEl.onplay = () => { playBtn.textContent = "⏸"; };
    audioEl.onpause = () => { playBtn.textContent = "▶"; };
    audioEl.onended = () => { playBtn.textContent = "▶"; };

    audioEl.onloadedmetadata = () => {
        timeDisplay.textContent = `0:00 / ${formatTime(audioEl.duration)}`;
    };

    audioEl.ontimeupdate = () => {
        const pct = audioEl.duration ? (audioEl.currentTime / audioEl.duration) * 100 : 0;
        progressFill.style.width = `${pct}%`;
        timeDisplay.textContent = `${formatTime(audioEl.currentTime)} / ${formatTime(audioEl.duration)}`;
    };

    // Seek on progress bar click
    progressContainer.onclick = (e) => {
        if (!audioEl.duration) return;
        const rect = progressContainer.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        audioEl.currentTime = pct * audioEl.duration;
    };

    // Initial preview
    updateAudioPreview(audioWidget.value);

    // Watch for dropdown value changes
    const origCallback = audioWidget.callback;
    audioWidget.callback = function () {
        const result = origCallback?.apply(this, arguments);
        updateAudioPreview(audioWidget.value);
        return result;
    };

    // --- Assemble DOM ---
    playerBar.appendChild(playBtn);
    playerBar.appendChild(timeDisplay);
    playerBar.appendChild(progressContainer);
    playerBar.appendChild(volIcon);

    wrapper.appendChild(loadBtn);
    wrapper.appendChild(playerBar);
    wrapper.appendChild(fileInput);
    wrapper.appendChild(audioEl);

    if (typeof node.addDOMWidget === "function") {
        node.addDOMWidget("bsai_upload", "button", wrapper, {
            getValue: () => audioWidget.value,
            setValue: (v) => { audioWidget.value = v; updateAudioPreview(v); },
        });
    } else {
        const checkAndAppend = () => {
            const el = node.element || node.dom;
            if (el) {
                const widgetsContainer = el.querySelector(".litegraph-widgets");
                if (widgetsContainer) {
                    widgetsContainer.appendChild(wrapper);
                } else {
                    el.appendChild(wrapper);
                }
            } else {
                requestAnimationFrame(checkAndAppend);
            }
        };
        requestAnimationFrame(checkAndAppend);
    }
}
