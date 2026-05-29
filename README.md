# AFL_public

`AFL_public` is a ComfyUI custom node pack maintained for AFL workflows.

It includes utility nodes for image and audio work, plus an **AFL Canvas PNG** export action that lets a ComfyUI App-mode workflow be packed into a single PNG for use in **AFL CANVAS** local mode.

## What this plugin includes

- image utility nodes
- audio utility nodes
- SeedVC-related nodes
- RVC-related wrapper nodes
- `Export AFL Canvas PNG` in the ComfyUI canvas right-click menu

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/recursionlaplace-eng/AFL_public.git
```

Install the base Python dependencies:

```bash
pip install -r requirements.txt
```

If you plan to use the RVC wrapper features, also install:

```bash
pip install -r requirements-rvc-wrapper.txt
```

Then restart ComfyUI.

## AFL Canvas PNG export

This repository adds a canvas-level export action for **AFL CANVAS** local Comfy apps.

In ComfyUI, right-click the canvas and use:

- `AFL Canvas`
  - `Export AFL Canvas PNG`

This exports a single PNG that AFL CANVAS can import directly, without separately providing:

- `app.json`
- `api.json`

## When to use it

Use this export when:

1. your workflow has already been configured in **ComfyUI App mode**
2. you want AFL CANVAS to recognize the workflow's exposed inputs and outputs
3. you want to move the app as a single PNG instead of managing multiple JSON files by hand

## What gets embedded in the PNG

The exported PNG includes:

- `workflow`
- `afl_app_json`
- `prompt`
- `afl_api_json`
- `afl_canvas_bundle`

It also embeds offline field definitions used by AFL CANVAS to reconstruct exposed controls more reliably when ComfyUI is not available at import time.

That is the key difference from a normal workflow PNG export.

## Important requirement

This export is intended for workflows already saved in **ComfyUI App mode**.

The workflow must contain App-mode exposure data, including:

- `extra.linearData.inputs`
- `extra.linearData.outputs`

If those fields are missing, export will stop and show a warning. That is expected, because AFL CANVAS would not know which inputs should be exposed in the imported app.

## Basic workflow

1. Build or open your workflow in ComfyUI.
2. Save/configure it in **App mode** and expose the inputs and outputs you want.
3. Right-click the canvas.
4. Choose `AFL Canvas` -> `Export AFL Canvas PNG`.
5. Import that PNG into AFL CANVAS local mode.

## Encoding and localized labels

The AFL Canvas metadata is written into UTF-8-safe PNG text chunks.

This is important for workflows that use:

- Chinese labels
- localized field names
- custom display names from App mode

Without that extra handling, localized text can become garbled during import.

## Dependency note

The AFL Canvas PNG export feature itself does **not** add any extra Python dependency.

It is implemented in the ComfyUI frontend extension layer under:

- `web/afl_canvas_export.js`

## Repository structure

- `nodes/` - ComfyUI node implementations
- `web/` - frontend extensions, including AFL Canvas export
- `docs/` - additional notes for audio, RVC, and SeedVC features
- `vendor/` - bundled helper code used by some nodes

## Extra docs

- `docs/AFL_RVC.md`
- `docs/AFL_SEED_VC.md`
- `docs/AFL_SEED_VC_MAINTENANCE.md`
- `docs/AFL_AUDIO_MEMORY_POLICY.md`
- `docs/AFL_LOCAL_AI_PORTABLE.md`

## Notes

Some features in this repository are self-contained, while others depend on local models or external runtimes that are documented in the corresponding files under `docs/`.

For open-source users, the safest starting point is:

1. install the plugin
2. verify ComfyUI loads without errors
3. test the `Export AFL Canvas PNG` menu first
4. then enable the heavier audio or voice-related features as needed
