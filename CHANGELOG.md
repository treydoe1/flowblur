# Changelog

## 0.5.1

First public version (alpha).

- Optical-flow motion blur effect for Adobe Premiere Pro on Apple Silicon Macs, with the
  parameters Blur Amount, Motion Sensitivity, Max Blur (%), Quality, Blend in Linear Light,
  View (Output / Motion Vectors) and Use Source Frame Rate.
- Neighbouring frames taken at the clip's source frame rate when it is clearly higher than the
  sequence's frame rate.
- Adjustment-layer support through Premiere Pro's GPU filter interface, using the same engine as
  the CPU effect path; `/tmp/flowblur-no-gpu` turns it off.
- Installer package, uninstall script, and an opt-in diagnostics log (`/tmp/flowblur-debug`).
