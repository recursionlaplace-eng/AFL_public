# AFL local AI portable layout

`AFL_public` audio nodes can use an optional `afl_local_ai` resource folder for
bundled Python packages and models. The lookup is relative to the plugin and the
current ComfyUI root. It does not depend on a drive letter or on the name of the
outer portable package folder.

Recommended layouts:

1. `ComfyUI/custom_nodes/AFL_public/afl_local_ai`
2. `ComfyUI/afl_local_ai`

Advanced override:

- Set `AFL_LOCAL_AI_RESOURCE_ROOT` to the absolute path of the `afl_local_ai`
  folder.

Legacy-compatible fallback:

- A sibling folder next to `ComfyUI`, such as `../afl_local_ai`, is still
  checked.

For distribution, prefer layout 1 or 2 so users can move the whole portable
ComfyUI folder anywhere without editing paths.
