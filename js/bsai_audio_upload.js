import { app } from "../../scripts/app.js";

/**
 * BSAI Audio Upload Extension
 *
 * Adds a file upload button and audio preview player to BSAI_IndexTTS2.5LoadAudio
 * nodes. ComfyUI's built-in upload widget only works for the core LoadAudio node
 * type, so we add our own for the BSAI custom node.
 *
 * Features:
 *   - "Choose File" button that opens a file picker
 *   - Audio player with play/pause and timeline
 *   - Auto-updates when dropdown selection changes
 */

const NODE_TYPE = "BSAI_IndexTTS2.5LoadAudio";

app.registerExtension({
    name: "BSAI.IndexTTS2.5.AudioUpload",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            // Wait for widgets to be initialized
            const audioWidget = this.widgets?.find((w) => w.name === "audio");
            if (!audioWidget) {
                // Retry after a tick if widgets aren't ready yet
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
    if (node._bsaiUploadSetup) return; // Prevent double-setup
    node._bsaiUploadSetup = true;

    // --- Build DOM container ---
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width: 100%; padding: 2px 0;";

    // Upload button
    const uploadBtn = document.createElement("button");
    uploadBtn.textContent = "📁 Choose Audio File";
    uploadBtn.style.cssText = `
        display: block; width: 100%; padding: 5px 8px; margin-bottom: 4px;
        background: #2a2a3e; color: #88aacc; border: 1px solid #3a5a7a;
        border-radius: 4px; cursor: pointer; font-size: 12px; text-align: center;
        transition: background 0.15s;
    `;
    uploadBtn.onmouseenter = () => { uploadBtn.style.background = "#3a3a4e"; };
    uploadBtn.onmouseleave = () => { uploadBtn.style.background = "#2a2a3e"; };

    // Hidden file input
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".wav,.mp3,.flac,.ogg,.m4a";
    fileInput.style.display = "none";

    uploadBtn.onclick = () => fileInput.click();

    fileInput.onchange = async () => {
        const file = fileInput.files[0];
        if (!file) return;

        uploadBtn.textContent = "⏳ Uploading...";
        uploadBtn.disabled = true;

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

            // Refresh the dropdown options if needed
            if (audioWidget.options && Array.isArray(audioWidget.options.values)) {
                if (!audioWidget.options.values.includes(data.name)) {
                    audioWidget.options.values.push(data.name);
                    audioWidget.options.values.sort();
                }
            }

            updateAudioPreview(data.name);
            uploadBtn.textContent = "📁 Choose Audio File";
        } catch (err) {
            console.error("[BSAI] Upload error:", err);
            alert(`Upload failed: ${err.message}`);
            uploadBtn.textContent = "📁 Choose Audio File";
        } finally {
            uploadBtn.disabled = false;
            fileInput.value = "";
        }
    };

    // Audio preview element
    const audioEl = document.createElement("audio");
    audioEl.controls = true;
    audioEl.preload = "metadata";
    audioEl.style.cssText = `
        display: none; width: 100%; margin-top: 2px;
        border-radius: 4px; height: 32px;
    `;

    function updateAudioPreview(filename) {
        if (filename && !filename.startsWith("upload_")) {
            audioEl.src = `/api/view?filename=${encodeURIComponent(filename)}&type=input`;
            audioEl.style.display = "block";
        } else {
            audioEl.src = "";
            audioEl.style.display = "none";
        }
    }

    // Initial preview
    updateAudioPreview(audioWidget.value);

    // Watch for dropdown value changes
    const origCallback = audioWidget.callback;
    audioWidget.callback = function () {
        const result = origCallback?.apply(this, arguments);
        updateAudioPreview(audioWidget.value);
        return result;
    };

    wrapper.appendChild(uploadBtn);
    wrapper.appendChild(fileInput);
    wrapper.appendChild(audioEl);

    // --- Add as DOM widget to the node ---
    if (typeof node.addDOMWidget === "function") {
        node.addDOMWidget("bsai_upload", "button", wrapper, {
            getValue: () => audioWidget.value,
            setValue: (v) => { audioWidget.value = v; updateAudioPreview(v); },
        });
    } else {
        // Fallback: append to node's DOM element directly
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
