import { app } from "../../scripts/app.js";

const MENU_LABEL = "AFL Canvas";
const EXPORT_LABEL = "Export AFL Canvas PNG";
const DOWNLOAD_NAME = "afl_canvas_bundle.png";
const EXTRA_PADDING = 100;

class AflCanvasPngExporter {
    static chunkType = {
        internationalText: "iTXt",
    };

    static encoder = new TextEncoder();

    n2b(value) {
        return new Uint8Array([
            (value >> 24) & 0xff,
            (value >> 16) & 0xff,
            (value >> 8) & 0xff,
            value & 0xff,
        ]);
    }

    joinArrayBuffer(...buffers) {
        const totalLength = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
        const result = new Uint8Array(totalLength);
        let offset = 0;
        for (const buffer of buffers) {
            result.set(buffer, offset);
            offset += buffer.byteLength;
        }
        return result;
    }

    crc32(data) {
        const crcTable =
            AflCanvasPngExporter.crcTable ||
            (AflCanvasPngExporter.crcTable = (() => {
                const table = [];
                for (let index = 0; index < 256; index += 1) {
                    let c = index;
                    for (let bit = 0; bit < 8; bit += 1) {
                        c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
                    }
                    table[index] = c;
                }
                return table;
            })());

        let crc = 0 ^ -1;
        for (let index = 0; index < data.byteLength; index += 1) {
            crc = (crc >>> 8) ^ crcTable[(crc ^ data[index]) & 0xff];
        }
        return (crc ^ -1) >>> 0;
    }

    buildChunk(type, data) {
        const typeBytes = AflCanvasPngExporter.encoder.encode(type);
        const chunkData = this.joinArrayBuffer(typeBytes, data);
        return this.joinArrayBuffer(
            this.n2b(data.byteLength),
            chunkData,
            this.n2b(this.crc32(chunkData)),
        );
    }

    buildInternationalTextChunk(key, value) {
        const keyword = AflCanvasPngExporter.encoder.encode(`${key}\0`);
        const compressionFlag = new Uint8Array([0]);
        const compressionMethod = new Uint8Array([0]);
        const languageTag = new Uint8Array([0]);
        const translatedKeyword = new Uint8Array([0]);
        const text = AflCanvasPngExporter.encoder.encode(String(value));
        const data = this.joinArrayBuffer(
            keyword,
            compressionFlag,
            compressionMethod,
            languageTag,
            translatedKeyword,
            text,
        );
        return this.buildChunk(AflCanvasPngExporter.chunkType.internationalText, data);
    }

    getBounds() {
        const bounds = app.graph._nodes.reduce(
            (acc, node) => {
                if (node.pos[0] < acc[0]) acc[0] = node.pos[0];
                if (node.pos[1] < acc[1]) acc[1] = node.pos[1];
                const nodeBounds = node.getBounding();
                const right = node.pos[0] + nodeBounds[2];
                const bottom = node.pos[1] + nodeBounds[3];
                if (right > acc[2]) acc[2] = right;
                if (bottom > acc[3]) acc[3] = bottom;
                return acc;
            },
            [99999, 99999, -99999, -99999]
        );

        bounds[0] -= EXTRA_PADDING;
        bounds[1] -= EXTRA_PADDING;
        bounds[2] += EXTRA_PADDING;
        bounds[3] += EXTRA_PADDING;
        return bounds;
    }

    saveCanvasState() {
        this.state = {
            scale: app.canvas.ds.scale,
            width: app.canvas.canvas.width,
            height: app.canvas.canvas.height,
            offset: app.canvas.ds.offset,
            transform: app.canvas.canvas.getContext("2d").getTransform(),
        };
    }

    restoreCanvasState() {
        app.canvas.ds.scale = this.state.scale;
        app.canvas.canvas.width = this.state.width;
        app.canvas.canvas.height = this.state.height;
        app.canvas.ds.offset = this.state.offset;
        app.canvas.canvas.getContext("2d").setTransform(this.state.transform);
    }

    updateView(bounds) {
        const scale = window.devicePixelRatio || 1;
        app.canvas.ds.scale = 1;
        app.canvas.canvas.width = (bounds[2] - bounds[0]) * scale;
        app.canvas.canvas.height = (bounds[3] - bounds[1]) * scale;
        app.canvas.ds.offset = [-bounds[0], -bounds[1]];
        app.canvas.canvas.getContext("2d").setTransform(scale, 0, 0, scale, 0, 0);
    }

    async collectExportPayload() {
        const promptResult = await app.graphToPrompt();
        const appJson = app.graph.serialize();
        const workflow = appJson;
        const prompt = promptResult?.output || promptResult?.prompt || null;

        if (!appJson?.extra?.linearData) {
            throw new Error("Current workflow does not contain App Mode exposure data (extra.linearData). Please save it in ComfyUI App mode first.");
        }

        return {
            workflow,
            appJson,
            prompt,
            bundleMeta: {
                format: "afl_canvas_comfy_bundle",
                version: 2,
                exportedAt: new Date().toISOString(),
                source: "AFL_public",
            },
        };
    }

    async renderCanvasBlob() {
        this.saveCanvasState();
        this.updateView(this.getBounds());
        app.canvas.draw(true, true);

        try {
            return await new Promise((resolve, reject) => {
                app.canvasEl.toBlob((blob) => {
                    if (!blob) {
                        reject(new Error("Failed to render canvas PNG."));
                        return;
                    }
                    resolve(blob);
                }, "image/png");
            });
        } finally {
            this.restoreCanvasState();
            app.canvas.draw(true, true);
        }
    }

    async embedMetadata(blob, payload) {
        const buffer = await blob.arrayBuffer();
        const typedArray = new Uint8Array(buffer);
        const view = new DataView(buffer);
        const ihdrSize = view.getUint32(8) + 20;

        const workflowChunk = this.buildInternationalTextChunk("workflow", JSON.stringify(payload.workflow));
        const appChunk = this.buildInternationalTextChunk("afl_app_json", JSON.stringify(payload.appJson));
        const promptChunk = payload.prompt ? this.buildInternationalTextChunk("prompt", JSON.stringify(payload.prompt)) : new Uint8Array();
        const apiChunk = payload.prompt ? this.buildInternationalTextChunk("afl_api_json", JSON.stringify(payload.prompt)) : new Uint8Array();
        const bundleChunk = this.buildInternationalTextChunk("afl_canvas_bundle", JSON.stringify(payload.bundleMeta));

        const result = this.joinArrayBuffer(
            typedArray.subarray(0, ihdrSize),
            workflowChunk,
            appChunk,
            promptChunk,
            apiChunk,
            bundleChunk,
            typedArray.subarray(ihdrSize),
        );

        return new Blob([result], { type: "image/png" });
    }

    download(blob, workflow) {
        const filenameBase = String(workflow?.extra?.workspaceName || workflow?.name || workflow?.title || "afl_canvas_bundle")
            .trim()
            .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, "_")
            .replace(/\s+/g, "_")
            .replace(/^_+|_+$/g, "");
        const downloadName = `${filenameBase || DOWNLOAD_NAME.replace(/\.png$/i, "")}.png`;

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        Object.assign(link, {
            href: url,
            download: downloadName,
            style: "display:none",
        });
        document.body.append(link);
        link.click();
        setTimeout(() => {
            link.remove();
            URL.revokeObjectURL(url);
        }, 0);
    }

    async export() {
        const payload = await this.collectExportPayload();
        const blob = await this.renderCanvasBlob();
        const bundledBlob = await this.embedMetadata(blob, payload);
        this.download(bundledBlob, payload.workflow);
    }
}

app.registerExtension({
    name: "AFL.Canvas.Export",
    setup() {
        const originalGetCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function (...args) {
            const options = originalGetCanvasMenuOptions.apply(this, args);
            options.push(null, {
                content: MENU_LABEL,
                submenu: {
                    options: [
                        {
                            content: EXPORT_LABEL,
                            callback: async () => {
                                try {
                                    const exporter = new AflCanvasPngExporter();
                                    await exporter.export();
                                } catch (error) {
                                    const message = error?.message || String(error || "Unknown export error.");
                                    console.error("[AFL Canvas Export]", error);
                                    alert(`AFL Canvas export failed:\n${message}`);
                                }
                            },
                        },
                    ],
                },
            });
            return options;
        };
    },
});
