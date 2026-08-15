/**
 * BSAI Audio Upload Extension
 *
 * Adds "加载音频" (Load Audio) and "播放音频" (Play Audio) buttons to
 * BSAI_IndexTTS2.5LoadAudio nodes.
 *
 * Features:
 *   - "加载音频" button that opens a file picker to upload audio
 *   - "播放音频" button that plays/pauses the selected audio
 *   - Auto-updates when dropdown selection changes
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

            const audioWidget = this.widgets?.find((w) => w.name === "audio");
            if (!audioWidget) {
                setTimeout(() => setupUpload(this), 0);
                return;
            }
            setupUpload(this);
        };
    },
});

function setupUpload(node) {
    const audioWidget = node.widgets?.find((w) => w.name === "audio");
    if (!audioWidget) return;
    if (node._bsaiUploadSetup) return;
    node._bsaiUploadSetup = true;

    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width: 100%; padding: 2px 0;";

    // --- Button row ---
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display: flex; gap: 4px; width: 100%;";

    // 加载音频 button
    const loadBtn = document.createElement("button");
    loadBtn.textContent = "📁 加载音频";
    loadBtn.style.cssText = `
        flex: 1; padding: 5px 8px;
        background: #2a2a3e; color: #88aacc; border: 1px solid #3a5a7a;
        border-radius: 4px; cursor: pointer; font-size: 12px; text-align: center;
        transition: background 0.15s;
    `;
    loadBtn.onmouseenter = () => { loadBtn.style.background = "#3a3a4e"; };
    loadBtn.onmouseleave = () => { loadBtn.style.background = "#2a2a3e"; };

    // 播放音频 button
    const playBtn = document.createElement("button");
    playBtn.textContent = "▶ 播放音频";
    playBtn.style.cssText = `
        flex: 1; padding: 5px 8px;
        background: #2a3e2a; color: #aacc88; border: 1px solid #5a7a3a;
        border-radius: 4px; cursor: pointer; font-size: 12px; text-align: center;
        transition: background 0.15s;
    `;
    playBtn.onmouseenter = () => { playBtn.style.background = "#3a4e3a"; };
    playBtn.onmouseleave = () => { playBtn.style.background = "#2a3e2a"; };
    playBtn.disabled = true;
    playBtn.style.opacity = "0.5";
    playBtn.style.cursor = "not-allowed";

    // Hidden file input
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
            loadBtn.style.opacity = "1";
            fileInput.value = "";
        }
    };

    // Audio element (hidden, controlled by play button)
    const audioEl = document.createElement("audio");
    audioEl.preload = "metadata";
    audioEl.style.cssText = "display: none;";

    function updateAudioPreview(filename) {
        if (filename && !filename.startsWith("upload_")) {
            audioEl.src = `/api/view?filename=${encodeURIComponent(filename)}&type=input`;
            playBtn.disabled = false;
            playBtn.style.opacity = "1";
            playBtn.style.cursor = "pointer";
            playBtn.textContent = "▶ 播放音频";
        } else {
            audioEl.src = "";
            audioEl.pause();
            playBtn.disabled = true;
            playBtn.style.opacity = "0.5";
            playBtn.style.cursor = "not-allowed";
            playBtn.textContent = "▶ 播放音频";
        }
    }

    // Play/pause toggle
    playBtn.onclick = () => {
        if (!audioEl.src) return;
        if (audioEl.paused) {
            audioEl.play().catch(err => {
                console.error("[BSAI] Playback error:", err);
            });
        } else {
            audioEl.pause();
        }
    };

    audioEl.onplay = () => { playBtn.textContent = "⏸ 停止播放"; };
    audioEl.onpause = () => { playBtn.textContent = "▶ 播放音频"; };
    audioEl.onended = () => { playBtn.textContent = "▶ 播放音频"; };

    // Initial preview
    updateAudioPreview(audioWidget.value);

    // Watch for dropdown value changes
    const origCallback = audioWidget.callback;
    audioWidget.callback = function () {
        const result = origCallback?.apply(this, arguments);
        updateAudioPreview(audioWidget.value);
        return result;
    };

    btnRow.appendChild(loadBtn);
    btnRow.appendChild(playBtn);
    wrapper.appendChild(btnRow);
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
