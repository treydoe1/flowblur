#!/bin/bash
# Builds FlowBlur and assembles the signed macOS plug-in bundle at ROOT/dist/FlowBlur.plugin:
#   Contents/MacOS/FlowBlur          the arm64 cdylib (exports EffectMain)
#   Contents/Resources/FlowBlur.rsrc the PiPL resource produced by build.rs (pipl crate)
#   Contents/PkgInfo, Contents/Info.plist (also from the pipl crate)
# then sets CFBundleIdentifier and ad-hoc codesigns it (Premiere/AE 25.2+ refuse unsigned plug-ins).
# Mirrors the create_bundle recipe of the after-effects bindings' AdobePlugin.just, arm64 only.
#
# This script installs NOTHING. To install, copy (a real copy, not an alias/symlink) the bundle to
#   "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"
# e.g.  sudo cp -R dist/FlowBlur.plugin "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/"
# and restart Premiere Pro (and After Effects, which also scans MediaCore).
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$PLUGIN_DIR/.." && pwd)"
CARGO="${CARGO:-cargo}"
command -v "$CARGO" >/dev/null 2>&1 || CARGO=/opt/homebrew/bin/cargo

NAME=FlowBlur
BIN=flowblur                       # Cargo package/lib name
BUNDLE_ID=com.flowblur.motionblur

cd "$PLUGIN_DIR"
# Keep local paths out of the shipped binary (panic-location strings). When several prefixes
# match, rustc applies the last one, so the more specific $ROOT mapping must come after $HOME.
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=~ --remap-path-prefix=$ROOT=flowblur"
"$CARGO" build --release

METADATA="$("$CARGO" metadata --format-version 1 --no-deps)"
TARGET_DIR="$(printf '%s' "$METADATA" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
# The version is written only in Cargo.toml; read it back for the bundle's version keys.
VERSION="$(printf '%s' "$METADATA" | /usr/bin/python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == sys.argv[1]))' "$BIN")"
[[ -n "$VERSION" ]] || { echo "could not read the package version from cargo metadata" >&2; exit 1; }
OUT="$TARGET_DIR/release"
# pipl writes its outputs to $OUT_DIR/../../.. . With cargo's classic layout that is
# target/release; with the newer build-dir layout (build/<pkg>/<hash>/out) it is
# target/release/build. Accept either, preferring the most recently written one.
RES=""
for d in "$OUT" "$OUT/build"; do
    if [[ -f "$d/$BIN.rsrc" && ( -z "$RES" || "$d/$BIN.rsrc" -nt "$RES/$BIN.rsrc" ) ]]; then RES="$d"; fi
done
[[ -n "$RES" ]] || { echo "missing $BIN.rsrc under $OUT" >&2; exit 1; }
for f in "$OUT/lib$BIN.dylib" "$RES/$BIN.rsrc" "$RES/${BIN}_PkgInfo" "$RES/${BIN}_Info.plist"; do
    [[ -f "$f" ]] || { echo "missing build output: $f" >&2; exit 1; }
done

BUNDLE="$ROOT/dist/$NAME.plugin"
echo "Creating $BUNDLE"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
# Plain copies (cp without -R never preserves symlinks for regular files), so the bundle is self-contained.
cp "$RES/$BIN.rsrc"          "$BUNDLE/Contents/Resources/$NAME.rsrc"
cp "$RES/${BIN}_PkgInfo"     "$BUNDLE/Contents/PkgInfo"
cp "$RES/${BIN}_Info.plist"  "$BUNDLE/Contents/Info.plist"
cp "$OUT/lib$BIN.dylib"      "$BUNDLE/Contents/MacOS/$NAME"

PLIST="$BUNDLE/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $BUNDLE_ID" "$PLIST"
# Standard keys Adobe's Xcode sample projects also carry (pipl only writes the three above).
/usr/libexec/PlistBuddy -c "Add :CFBundleExecutable string $NAME" \
                        -c "Add :CFBundleName string $NAME" \
                        -c "Add :CFBundleShortVersionString string $VERSION" \
                        -c "Add :CFBundleVersion string $VERSION" \
                        -c "Add :CFBundleInfoDictionaryVersion string 6.0" "$PLIST"

# Same flags as the reference recipe; ad-hoc identity unless an Apple Development cert exists.
DEV_CERT="$(security find-identity -v -p codesigning 2>/dev/null | grep -m 1 "Apple Development" | awk '{print $2}' || true)"
if [[ -n "$DEV_CERT" ]]; then
    codesign --force --options runtime --timestamp --strict --sign "$DEV_CERT" "$BUNDLE"
else
    codesign --force --options runtime --strict --sign - "$BUNDLE"
    echo "Note: ad-hoc signature. For distribution use a Developer ID certificate and notarize."
fi
codesign --verify --deep --strict "$BUNDLE"

if find "$BUNDLE" -type l | grep -q .; then
    echo "error: bundle contains symlinks" >&2; exit 1
fi

# Verify both entry points are exported.
EXPORTS="$(nm -gU "$BUNDLE/Contents/MacOS/$NAME")"
echo "$EXPORTS" | grep -q "_EffectMain"      || { echo "error: EffectMain not exported" >&2; exit 1; }
echo "$EXPORTS" | grep -q "_xGPUFilterEntry" || { echo "error: xGPUFilterEntry not exported" >&2; exit 1; }

echo "Built and signed: $BUNDLE"
