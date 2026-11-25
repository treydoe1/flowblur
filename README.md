# FlowBlur

FlowBlur is an open-source motion blur effect for Adobe Premiere Pro. It estimates the motion
in each frame with dense optical flow and blurs every pixel along its own motion, so moving
objects get streaks that follow how they actually move.

This is an alpha release (version 0.6.0). Expect rough edges, and keep backups of projects you
use it in.

## Requirements

- A Mac with Apple Silicon. Intel Macs are not supported.
- macOS 11 or later.
- Adobe Premiere Pro. FlowBlur has been tested only with Premiere Pro 2027.

## Install

1. Open the release page, <https://github.com/treydoe1/flowblur/releases>, and download the
   `FlowBlur-<version>.pkg` attached to the latest release.
2. Quit Premiere Pro and After Effects. They load plug-ins only when they start.
3. Double-click the package.

   The package is not signed with an Apple Developer ID YET (coming in a few days), so macOS will refuse to open it the
   first time and say it cannot verify the developer. To open it anyway:

   1. Click Done (or OK) in that message.
   2. Open System Settings, then Privacy & Security.
   3. Scroll down to the Security section. There is a line saying that "FlowBlur-0.6.0.pkg" was
      blocked. Click Open Anyway and confirm with your password or Touch ID.
   4. The installer starts. Follow its steps.

4. The installer copies `FlowBlur.plugin` to
   `/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/`, replacing any earlier
   copy there.
5. Start Premiere Pro. The effect is in the Effects panel under Video Effects, FlowBlur.

### Uninstall

Quit Premiere Pro and After Effects, then run `packaging/uninstall.sh` from this repository as
your normal user:

```sh
./packaging/uninstall.sh
```

It asks for your password (through `sudo`), removes
`/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/FlowBlur.plugin`, and removes
the installer receipt. It refuses to delete that path if the bundle there is not FlowBlur. It
leaves the update-check folder `~/Library/Application Support/FlowBlur` in place and prints how
to remove it.

## Build from source

You need:

- a Mac with Apple Silicon,
- the Xcode Command Line Tools (`xcode-select --install`), which provide `codesign`, `pkgbuild`,
  `productbuild` and the other tools the scripts use,
- a recent stable Rust toolchain (the crates use Rust edition 2024), for example from
  <https://rustup.rs>.

The first build downloads the Rust dependencies, including the Adobe plug-in bindings from
GitHub.

Get the source:

```sh
git clone https://github.com/treydoe1/flowblur.git
cd flowblur
```

Build the plug-in bundle:

```sh
plugin/build.sh
```

This compiles the effect in release mode and writes a code-signed bundle to
`dist/FlowBlur.plugin` at the top of the repository. It uses a "Developer ID Application" or
else an "Apple Development" signing identity if your keychain has one, and an ad-hoc signature
otherwise (Premiere Pro refuses unsigned plug-ins). The script installs nothing.

Build the installer package:

```sh
packaging/build_pkg.sh            # runs plugin/build.sh first
packaging/build_pkg.sh --no-build # packages the existing dist/FlowBlur.plugin
```

This writes `dist/FlowBlur-<version>.pkg`. The package is signed only if a "Developer ID
Installer" identity is in your keychain, and it is notarized only with `--notarize` (see below).

To install a bundle you built without making a package, copy it (a real copy, not an alias or
symlink) into the plug-in folder and restart Premiere Pro:

```sh
sudo cp -R dist/FlowBlur.plugin "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"
```

Run the tests with `cargo test --release` in `core/` and in `plugin/`.

The repository has three parts:

- `core/`: the image-processing engine (optical flow and blur). It has no Adobe dependencies and
  includes a small command-line tool, `flowblur-cli`, that processes PNG files.
- `plugin/`: the Premiere Pro effect, which reads frames from the host, calls the engine and
  writes the result.
- `packaging/`: the installer package build, its preinstall script, and the uninstall script.

### Signed and notarized release

A package that opens with a normal double-click needs an Apple Developer Program membership
with two certificates in your keychain, "Developer ID Application" (for the plug-in bundle)
and "Developer ID Installer" (for the package), and Apple's notary service.

Once, store your notary service credentials in the keychain under a profile name of your
choice (it asks for your Apple ID, team ID and an app-specific password; an App Store Connect
API key works too, see `xcrun notarytool store-credentials --help`):

```sh
xcrun notarytool store-credentials flowblur-notary
```

Then build, sign, notarize and staple in one step:

```sh
packaging/build_pkg.sh --notarize flowblur-notary
```

Before uploading anything, the script checks that the package is signed with a Developer ID
Installer identity and that the bundle in it is signed with a Developer ID Application identity,
the hardened runtime and a secure timestamp, and names any requirement that is missing. It fails
if Apple does not accept the package (and prints Apple's log), or if stapling or the Gatekeeper
check fails.

Settings, as environment variables:

- `FLOWBLUR_SIGN_IDENTITY`: the identity `plugin/build.sh` signs the bundle with, as a name or
  SHA-1 hash (`security find-identity -v -p codesigning` lists them); `-` forces an ad-hoc
  signature. Without it, the first Developer ID Application identity is used, else the first
  Apple Development identity, else ad-hoc.
- `FLOWBLUR_INSTALLER_IDENTITY`: the name of the identity the package is signed with; `-`
  leaves it unsigned. Without it, the first Developer ID Installer identity is used, else none.
- `FLOWBLUR_NOTARY_PROFILE`: the same as `--notarize <profile>`.

A Developer ID signature carries the certificate holder's name and team ID, and anyone can read
them from the package and the bundle (`pkgutil --check-signature`, `codesign -dvv`).

## Parameters

**Blur Amount** (default 0.50, range 0 to 4): how long the virtual shutter stays open, as a
fraction of one frame interval. 0.5 corresponds to a 180-degree shutter and 1.0 to a 360-degree
shutter. 0 turns the blur off and passes the frame through unchanged.

**Motion Sensitivity** (default 70, range 0 to 100): how much of the detected motion is trusted.
Lower values smooth the motion field more, ignore motion the flow is less sure about, and ignore
small movements, so only clear, larger motion is blurred. Higher values also blur finer and
smaller motion. Lowering it never adds blur; 0 turns the blur off.

**Max Blur (%)** (default 15, range 0 to 50): the longest blur streak allowed, as a percentage
of the frame's longer side. Longer streaks are shortened to this length.

**Quality** (Draft, Normal, High; default Normal): the resolution at which motion is analysed
(a longer side of 384, 640 or 1024 pixels), how much work the optical flow does, and how many
samples the blur takes along each streak. Higher settings are slower.

**Blend in Linear Light** (default on): averages colours in linear light rather than on the
gamma-encoded values, which keeps bright highlights bright when they are smeared. Turn it off to
blend the encoded values directly.

**View** (Output or Motion Vectors; default Output): Output shows the blurred frame. Motion
Vectors shows the motion field the blur uses as a colour image: the hue is the direction of
motion, the saturation is its speed, and white means no motion. Use it to see what FlowBlur
detects.

**Use Source Frame Rate** (default on; Premiere Pro only): when a clip's own frame rate is
clearly higher than the sequence's (for example 119.88 fps footage in a 23.976 fps sequence),
FlowBlur compares the current frame with the clip's directly adjacent source frames instead of
the frames one sequence frame away, and scales the blur to match. Without this, such footage
would be compared with frames several source frames apart. Off, or when the
clip does not qualify (including adjustment layers, nested sequences and footage at or below the
sequence rate), the neighbouring frames are taken one sequence frame apart.

**Update notice** (a button, always last): the row is named `FlowBlur <version>` and the button
Release Notes, or, when the update check has found a newer release, the row is named "Update
available" and the button is named after that version. Clicking it opens the release page
in your browser and does nothing else. It does not affect rendering and cannot be keyframed.

## How it works

For each frame, FlowBlur converts the current frame and its neighbouring frames (the previous
and the next) to luma at the analysis resolution and estimates dense optical flow between them.
It removes unreliable vectors, rejects scene cuts, and smooths the result into one motion vector
per pixel. It then scales that field up to full resolution and renders the blur by gathering
samples along each pixel's motion.

When FlowBlur is applied to an adjustment layer, the neighbouring frames are the composite of
the layers below, and they are fetched through Premiere Pro's GPU filter interface. The frames
are processed by the same engine as on the CPU path. Everywhere else, FlowBlur uses the standard
CPU effect path.

## Update check

When Premiere Pro or After Effects loads the plug-in, FlowBlur checks at most once every 24
hours whether a newer release exists. The same happens in any other program that loads the
plug-in, for example Adobe Media Encoder; the shared check time usually keeps them from all
checking.

- It makes one HTTPS request to github.com, asking for
  `https://github.com/treydoe1/flowblur/releases/latest`, and reads only the address GitHub
  redirects to, which names the latest release's version. The request runs in the system
  `curl` program in the background, so loading the plug-in does not wait for it.
  A network monitor therefore shows the request as coming from `curl`. The request is made
  directly, so it does not go through a configured proxy; where direct connections are
  blocked, the check finds nothing.
- GitHub can see your IP address and the client name `FlowBlur/<version>`. Nothing about your
  projects, footage or computer is sent.
- Nothing is downloaded or installed. When the release is newer than the installed version,
  the About text and the last parameter row say so; installing it is up to you.
- The time of the last check and the latest version seen are kept in
  `~/Library/Application Support/FlowBlur/update-check`. The About text shows a newer release
  as soon as a check has found it, even in the same session; the parameter row changes the
  next time the plug-in is loaded.

To turn the check off (no request and no notice):

```sh
mkdir -p ~/Library/Application\ Support/FlowBlur && touch ~/Library/Application\ Support/FlowBlur/no-update-check
```

To turn it on again:

```sh
rm ~/Library/Application\ Support/FlowBlur/no-update-check
```

## Switches

Three files act as switches. Create them with `touch` and delete them to switch back. The two in
`/tmp` take effect on the next rendered frame, without restarting Premiere Pro; `/tmp` is
emptied when the Mac restarts.

- `/tmp/flowblur-no-gpu`: turns the GPU path off, so adjustment layers also use the CPU path.
- `/tmp/flowblur-debug`: writes one line per rendered frame to `/tmp/flowblur-debug.log`. The
  file must be a regular file owned by you and not writable by other users, which is what
  `touch /tmp/flowblur-debug` creates.
- `~/Library/Application Support/FlowBlur/no-update-check`: turns the update check off (see
  Update check). It is read each time the plug-in is loaded and stays across restarts.

## Known limitations

- Apple Silicon Macs only.
- Tested only with Premiere Pro 2027.
- Clips that run slightly slower than the timeline (for example 23.976 fps footage on a 25 fps
  timeline) can show the blur strength changing about once a second.
- Slowed-down high-frame-rate clips that use Frame Blending or Optical Flow time interpolation
  get too much blur.
- Speed-changed nested sequences have not been tested.
- After Effects loads the effect, but it has not been tested there.

## Licence

FlowBlur is dual-licensed under either of

- the MIT licence (`LICENSE-MIT`), or
- the Apache License, Version 2.0 (`LICENSE-APACHE`),

at your option.

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion
in FlowBlur is accepted under the same terms, dual-licensed as above, without any additional
terms or conditions.

## Trademarks

FlowBlur is not affiliated with or endorsed by Adobe. Adobe, Premiere Pro and After Effects are
either registered trademarks or trademarks of Adobe in the United States and/or other countries.
