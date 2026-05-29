# AFL Seed VC

`AFL Seed VC` is the AFL_public maintained SeedVC node.

## Nodes

- `AFL Seed VC`: voice conversion with seed control, short reference prompt handling, and chunked source processing.
- `AFL Match Audio Loudness`: matches one audio signal to the loudness of another audio signal.

## Suggested Workflow

1. Normalize source and reference audio before `AFL Seed VC`.
2. Run `AFL Seed VC`.
3. Feed the converted output into `AFL Match Audio Loudness`.
4. Use the original source audio as `reference_audio` if you want the converted result to land near the original input volume.

## Seed

The seed controls the diffusion noise used by SeedVC. The same seed should be repeatable. Different seeds usually change small voice details such as breath, texture, tail sounds, or local stability; it will not rewrite the content like a text-to-image seed.

## F0 / Pitch

Enable `f0_condition` for singing or pitch-sensitive material. `auto_f0_adjust` aligns the source pitch range toward the reference pitch range. `pitch_shift` then shifts the converted pitch in semitones.
