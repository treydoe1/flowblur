#!/bin/bash
# Builds the FlowBlur macOS installer package: ROOT/dist/FlowBlur-<version>.pkg
#
#   packaging/build_pkg.sh             run plugin/build.sh first, then package dist/FlowBlur.plugin
#   packaging/build_pkg.sh --no-build  package the existing dist/FlowBlur.plugin as is
#
# The package installs FlowBlur.plugin (root:wheel, real files, no symlinks) to
#   /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/
# It is non-relocatable, so Installer never "upgrades" some other copy of the bundle found on disk.
# Signed with a "Developer ID Installer" identity if one is in the keychain, otherwise unsigned.
# Not notarized. This script installs nothing and needs no sudo.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$PKG_DIR/.." && pwd)"
NAME=FlowBlur
BUNDLE_ID=com.flowblur.motionblur
PKG_ID=com.flowblur.motionblur.pkg
INSTALL_LOCATION="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
BUNDLE="$ROOT/dist/$NAME.plugin"
BINARY="$BUNDLE/Contents/MacOS/$NAME"

BUILD=1
for arg in "$@"; do
    case "$arg" in
        --no-build) BUILD=0 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }

# 1. Build (or reuse) the bundle and verify its signature.
if [[ $BUILD == 1 ]]; then
    "$ROOT/plugin/build.sh"
fi
[[ -d "$BUNDLE" ]] || die "$BUNDLE not found (run without --no-build)"
codesign --verify --deep --strict "$BUNDLE" || die "code signature of $BUNDLE does not verify"
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$BUNDLE/Contents/Info.plist")" == "$BUNDLE_ID" ]] \
    || die "bundle identifier is not $BUNDLE_ID"
[[ "$(lipo -archs "$BINARY")" == "arm64" ]] || die "binary is not arm64-only"

# 2. Version: single source is plugin/Cargo.toml, read the same way build.sh does.
CARGO="${CARGO:-cargo}"
command -v "$CARGO" >/dev/null 2>&1 || CARGO=/opt/homebrew/bin/cargo
VERSION="$(cd "$ROOT/plugin" && "$CARGO" metadata --format-version 1 --no-deps \
    | /usr/bin/python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "flowblur"))')"
[[ -n "$VERSION" ]] || die "could not read the version from cargo metadata"
BUNDLE_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$BUNDLE/Contents/Info.plist")"
[[ "$BUNDLE_VERSION" == "$VERSION" ]] \
    || die "bundle version $BUNDLE_VERSION != Cargo.toml version $VERSION (rebuild without --no-build)"

# 3. Minimum macOS: the binary's own deployment target (LC_BUILD_VERSION minos, or LC_VERSION_MIN_MACOSX).
MIN_OS="$(otool -l "$BINARY" | awk '
    /cmd LC_BUILD_VERSION/      { b=1 }
    b && $1=="minos"            { print $2; exit }
    /cmd LC_VERSION_MIN_MACOSX/ { m=1 }
    m && $1=="version"          { print $2; exit }')"
[[ "$MIN_OS" =~ ^[0-9]+(\.[0-9]+)*$ ]] || die "could not read the deployment target from $BINARY"

OUT="$ROOT/dist/$NAME-$VERSION.pkg"
echo "Packaging $NAME $VERSION (minimum macOS $MIN_OS) -> $OUT"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/flowblur-pkg.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Payload root: a clean copy of the bundle without extended attributes, resource forks or ACLs.
export COPYFILE_DISABLE=1
mkdir -p "$WORK/root" "$WORK/scripts" "$WORK/pkgs"
ditto --norsrc --noextattr --noacl "$BUNDLE" "$WORK/root/$NAME.plugin"
codesign --verify --deep --strict "$WORK/root/$NAME.plugin" || die "signature broke while staging"
if find "$WORK/root" -type l -o -name .DS_Store -o -name '._*' | grep -q .; then
    die "staged payload contains symlinks, .DS_Store or ._ files"
fi

cp "$PKG_DIR/scripts/preinstall" "$WORK/scripts/preinstall"
chmod 755 "$WORK/scripts/preinstall"

# Component plist: start from pkgbuild's analysis, then pin the bundle to the install location.
PLIST="$WORK/component.plist"
pkgbuild --analyze --root "$WORK/root" "$PLIST" >/dev/null
[[ "$(/usr/libexec/PlistBuddy -c 'Print :0:RootRelativeBundlePath' "$PLIST")" == "$NAME.plugin" ]] \
    || die "unexpected component plist layout"
/usr/libexec/PlistBuddy -c 'Delete :0:ChildBundles' "$PLIST" 2>/dev/null || true
for kv in "BundleIsRelocatable:false" "BundleIsVersionChecked:false" \
          "BundleHasStrictIdentifier:true" "BundleOverwriteAction:upgrade"; do
    key="${kv%%:*}"; val="${kv#*:}"
    type=bool; [[ "$key" == BundleOverwriteAction ]] && type=string
    /usr/libexec/PlistBuddy -c "Delete :0:$key" "$PLIST" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :0:$key $type $val" "$PLIST"
done

pkgbuild --root "$WORK/root" \
         --component-plist "$PLIST" \
         --identifier "$PKG_ID" \
         --version "$VERSION" \
         --install-location "$INSTALL_LOCATION" \
         --scripts "$WORK/scripts" \
         --ownership recommended \
         --filter '/\.DS_Store$' --filter '/\._' \
         "$WORK/component-raw.pkg"

# pkgbuild archives every extended attribute as an AppleDouble "._name" entry, and macOS can
# attach com.apple.provenance to freshly written files in a way that cannot be removed. Rebuild
# the component's Payload, Scripts and Bom without those entries (same files, modes and root:wheel
# ownership pkgbuild recorded), then re-wrap it as a flat component with xar. (pkgutil --flatten
# would re-archive the Scripts directory and bring the "._preinstall" entry back.) xar's
# --distribution keeps only name, type, mode and data, so the archive records neither extended
# attributes nor the build user's name and uid.
C="$WORK/component"
pkgutil --expand "$WORK/component-raw.pkg" "$C"
lsbom "$C/Bom" | grep -v '/\._' > "$WORK/bom.txt"
mkbom -i "$WORK/bom.txt" "$C/Bom.new" && mv "$C/Bom.new" "$C/Bom"
(cd "$WORK/root" && find . | LC_ALL=C sort | cpio -o --format odc -R 0:0 2>/dev/null) | gzip -9 > "$C/Payload"
rm -rf "$C/Scripts"
(cd "$WORK/scripts" && find . | LC_ALL=C sort | cpio -o --format odc -R 0:0 2>/dev/null) | gzip -9 > "$C/Scripts"
NFILES="$(wc -l < "$WORK/bom.txt" | tr -d ' ')"
/usr/bin/sed -i '' -E "s/numberOfFiles=\"[0-9]+\"/numberOfFiles=\"$NFILES\"/" "$C/PackageInfo"
(cd "$C" && xar --compression none --distribution -cf "$WORK/pkgs/$NAME-component.pkg" \
    Bom Payload Scripts PackageInfo)

# Distribution: title, welcome text, Apple Silicon only, minimum macOS, system domain only.
cat > "$WORK/distribution.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>FlowBlur</title>
    <welcome file="welcome.txt" mime-type="text/plain"/>
    <options customize="never" require-scripts="false" hostArchitectures="arm64"/>
    <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
    <allowed-os-versions>
        <os-version min="$MIN_OS"/>
    </allowed-os-versions>
    <installation-check script="flowblur_check()"/>
    <script><![CDATA[
function flowblur_check() {
    if (system.sysctl('hw.optional.arm64') != 1) {
        my.result.type = 'Fatal';
        my.result.title = 'Apple Silicon required';
        my.result.message = 'FlowBlur requires a Mac with Apple Silicon. It cannot be installed on an Intel Mac.';
        return false;
    }
    return true;
}
    ]]></script>
    <pkg-ref id="$PKG_ID" version="$VERSION" onConclusion="none">$NAME-component.pkg</pkg-ref>
    <choices-outline>
        <line choice="default">
            <line choice="$PKG_ID"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="$PKG_ID" visible="false">
        <pkg-ref id="$PKG_ID"/>
    </choice>
</installer-gui-script>
EOF

SIGN_ARGS=()
INSTALLER_ID="$(security find-identity -v 2>/dev/null | awk -F'"' '/Developer ID Installer/ { print $2; exit }' || true)"
if [[ -n "$INSTALLER_ID" ]]; then
    SIGN_ARGS=(--sign "$INSTALLER_ID" --timestamp)
fi

rm -f "$OUT"
productbuild --distribution "$WORK/distribution.xml" \
             --resources "$PKG_DIR/resources" \
             --package-path "$WORK/pkgs" \
             "${SIGN_ARGS[@]+"${SIGN_ARGS[@]}"}" \
             "$OUT"

# Verify the built package: install location, non-relocatable bundle, identical signed payload.
X="$WORK/expanded"
pkgutil --expand-full "$OUT" "$X"
INFO="$X/$NAME-component.pkg/PackageInfo"
PAYLOAD="$X/$NAME-component.pkg/Payload"
grep -q "install-location=\"$INSTALL_LOCATION\"" "$INFO" || die "PackageInfo install-location is wrong"
grep -q "identifier=\"$PKG_ID\"" "$INFO" && grep -q "version=\"$VERSION\"" "$INFO" || die "PackageInfo id/version wrong"
grep -q "<relocate>" "$INFO" && die "PackageInfo still has a <relocate> section (bundle is relocatable)"
cmp "$PAYLOAD/$NAME.plugin/Contents/MacOS/$NAME" "$BINARY" || die "payload binary differs from $BINARY"
codesign --verify --deep --strict "$PAYLOAD/$NAME.plugin" || die "payload signature does not verify"
if find "$PAYLOAD" -type l -o -name .DS_Store -o -name '._*' | grep -q .; then
    die "payload contains symlinks, .DS_Store or ._ files"
fi
if { pkgutil --payload-files "$OUT"; lsbom -s "$X/$NAME-component.pkg/Bom"; } | grep -E '(^|/)(\._|\.DS_Store$)'; then
    die "package payload or Bom lists ._ or .DS_Store entries"
fi
[[ "$(ls -A "$X/$NAME-component.pkg/Scripts")" == "preinstall" ]] \
    || die "Scripts should contain only preinstall: $(ls -A "$X/$NAME-component.pkg/Scripts" | tr '\n' ' ')"

if [[ -n "$INSTALLER_ID" ]]; then
    echo "Signed with: $INSTALLER_ID (not notarized)"
else
    echo "Note: no \"Developer ID Installer\" identity in the keychain, so the package is UNSIGNED."
    echo "      Gatekeeper will block a double-click; see the README for how to open it."
fi
echo "Built and verified: $OUT"
