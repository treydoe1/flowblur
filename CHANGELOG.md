# Changelog

## 0.6.0

- Update notice: at most once every 24 hours, when the host loads the plug-in, FlowBlur asks
  GitHub for the version of the latest release. When it is newer, the About text says so and
  the last parameter row reads "Update available" with a button that opens the release page.
  Nothing is downloaded or installed. The file
  `~/Library/Application Support/FlowBlur/no-update-check` turns the check off.
- New last parameter: a button that opens the release page (row name `FlowBlur <version>`, or
  "Update available"). The seven existing parameters are unchanged, so existing projects keep
  their settings.

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
