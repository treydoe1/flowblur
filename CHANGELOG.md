# Changelog

## Unreleased

- Moving edges keep their motion. The forward-backward check in the motion analysis also tries the
  pixels around each landing point, so correct motion on both sides of a motion edge is no longer
  rejected and the edge is blurred instead of left sharp.

## 0.7.0

- FlowBlur is now called T Blur. The effect is under Video Effects, T Blur. The repository moved
  to <https://github.com/treydoe1/tblur>.
- The installer installs `TBlur.plugin` and removes an installed `FlowBlur.plugin` and its
  installer receipt, so Premiere Pro does not list both effects. `packaging/uninstall.sh` removes
  either one.
- The effect has a new match name, `TBLUR Motion Blur`. Projects saved with FlowBlur show the
  effect as missing. Apply T Blur to those clips again.
- New names for the files and settings that had FlowBlur in them: the switches are
  `/tmp/tblur-no-gpu` and `/tmp/tblur-debug`, the update check keeps its state in
  `~/Library/Application Support/TBlur`, and the build settings are `TBLUR_SIGN_IDENTITY`,
  `TBLUR_INSTALLER_IDENTITY` and `TBLUR_NOTARY_PROFILE`.
- FlowBlur 0.6.0 does not report this release as an update. Its update check accepts only the old
  repository address.

## 0.6.0

- Update notice: at most once every 24 hours, when the host loads the plug-in, FlowBlur asks
  GitHub for the version of the latest release. When it is newer, the About text says so and
  the last parameter row reads "Update available" with a button that opens the release page.
  Nothing is downloaded or installed. The file
  `~/Library/Application Support/FlowBlur/no-update-check` turns the check off.
- New last parameter: a button that opens the release page (row name `FlowBlur <version>`, or
  "Update available"). The seven existing parameters are unchanged, so existing projects keep
  their settings.
- Build scripts: the plug-in and installer package can be signed with an Apple Developer ID,
  and `packaging/build_pkg.sh --notarize <profile>` notarizes and staples the package.

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
