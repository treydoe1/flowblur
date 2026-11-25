#!/bin/bash
# Builds the FlowBlur macOS installer package: ROOT/dist/FlowBlur-<version>.pkg
#
#   packaging/build_pkg.sh                     run plugin/build.sh first, then package dist/FlowBlur.plugin
#   packaging/build_pkg.sh --no-build          package the existing dist/FlowBlur.plugin as is
#   packaging/build_pkg.sh --notarize PROFILE  also notarize and staple the package, using the
#                                              notarytool keychain profile PROFILE (or set
#                                              FLOWBLUR_NOTARY_PROFILE)
#
# The package installs FlowBlur.plugin (root:wheel, real files, no symlinks) to
#   /Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/
# It is non-relocatable, so Installer never "upgrades" some other copy of the bundle found on disk.
# Signed with FLOWBLUR_INSTALLER_IDENTITY if set ("-" means unsigned), else with the first
# "Developer ID Installer" identity in the keychain, else unsigned. Notarized only with --notarize,
# which requires a Developer ID Installer signature and a Developer ID Application bundle signature
# with the hardened runtime and a secure timestamp. This script installs nothing and needs no sudo.
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
NOTARY_PROFILE="${FLOWBLUR_NOTARY_PROFILE:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --notarize)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] \
                || { echo "--notarize needs the name of a notarytool keychain profile" >&2; exit 2; }
            NOTARY_PROFILE="$2"; shift ;;
        --notarize=*)
            NOTARY_PROFILE="${1#--notarize=}"
            [[ -n "$NOTARY_PROFILE" ]] \
                || { echo "--notarize needs the name of a notarytool keychain profile" >&2; exit 2; } ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
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

# Not named flowblur-*: an installed FlowBlur reads /tmp/flowblur-* names as switches.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/fb-pkg.XXXXXX")"
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

# Installer identity: FLOWBLUR_INSTALLER_IDENTITY (the certificate's name, passed to productbuild
# as is; "-" means unsigned), else the first valid "Developer ID Installer" identity, else unsigned.
# A given identity that productbuild cannot use is an error, never a silent fallback.
SIGN_ARGS=()
INSTALLER_ID="${FLOWBLUR_INSTALLER_IDENTITY:-}"
if [[ -z "$INSTALLER_ID" ]]; then
    INSTALLER_ID="$(security find-identity -v 2>/dev/null \
        | awk -F'"' 'index($2, "Developer ID Installer: ") == 1 && !n++ { print $2 }' || true)"
fi
[[ "$INSTALLER_ID" == "-" ]] && INSTALLER_ID=""
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

# What the package and the bundle inside it are actually signed with. The leaf certificate's
# name has the form "<kind>: <holder> (<team id>)"; only the kind is printed below.
# (pkgutil exits non-zero for an unsigned or invalid signature; "Timestamp=" is codesign's secure
# timestamp, "Signed Time=" an untrusted one.)
PKG_SIGNER=""
if PKG_SIG="$(pkgutil --check-signature "$OUT" 2>&1)"; then
    PKG_SIGNER="$(awk '/^ *1\. / && !n++ { sub(/^ *1\. /, ""); print }' <<<"$PKG_SIG")"
fi
APP_SIG="$(codesign -dvv "$PAYLOAD/$NAME.plugin" 2>&1 || true)"
APP_AUTHORITY="$(awk -F= '$1 == "Authority" && !n++ { print $2 }' <<<"$APP_SIG")"
APP_RUNTIME=0
grep -Eq '^CodeDirectory .*flags=0x[0-9a-fA-F]+\(([^)]*,)?runtime(,[^)]*)?\)' <<<"$APP_SIG" && APP_RUNTIME=1
APP_TIMESTAMP=0
grep -q '^Timestamp=' <<<"$APP_SIG" && APP_TIMESTAMP=1
PKG_KIND="${PKG_SIGNER%%:*}"
APP_KIND="${APP_AUTHORITY%%:*}"

NOTARIZED=0
if [[ -n "$NOTARY_PROFILE" ]]; then
    # 4. Notarize. Check every requirement first, so nothing is uploaded that must be rejected.
    MISSING=()
    [[ "$PKG_SIGNER" == "Developer ID Installer: "* ]] \
        || MISSING+=("the package is not signed with a Developer ID Installer identity (it is ${PKG_KIND:-unsigned})")
    [[ "$APP_AUTHORITY" == "Developer ID Application: "* ]] \
        || MISSING+=("$NAME.plugin is not signed with a Developer ID Application identity (it is ${APP_KIND:-ad-hoc})")
    [[ $APP_RUNTIME == 1 ]] \
        || MISSING+=("$NAME.plugin is not signed with the hardened runtime (codesign --options runtime)")
    [[ $APP_TIMESTAMP == 1 ]] \
        || MISSING+=("$NAME.plugin has no secure timestamp (codesign --timestamp)")
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "error: not submitting $OUT for notarization:" >&2
        printf '  - %s\n' "${MISSING[@]}" >&2
        echo "  See \"Signed and notarized release\" in the README." >&2
        exit 1
    fi

    echo "Submitting $OUT for notarization (profile \"$NOTARY_PROFILE\"); this can take several minutes."
    NOTARY_OUT="$WORK/notary.json"
    rc=0
    xcrun notarytool submit "$OUT" --keychain-profile "$NOTARY_PROFILE" --wait --output-format json \
        > "$NOTARY_OUT" || rc=$?
    # notarytool prints one JSON object at the end; read its "id" and "status". Anything else
    # (no JSON, a timeout, a status other than Accepted) is a failure.
    json_field() {
        /usr/bin/python3 -c '
import json, sys
try:
    v = json.load(open(sys.argv[1])).get(sys.argv[2], "")
except Exception:
    v = ""
print(v if isinstance(v, str) else "")' "$NOTARY_OUT" "$1"
    }
    SUB_ID="$(json_field id)"
    SUB_STATUS="$(json_field status)"
    if [[ $rc -ne 0 || "$SUB_STATUS" != "Accepted" ]]; then
        echo "error: notarization failed (notarytool exit status $rc, submission status \"${SUB_STATUS:-none}\")" >&2
        [[ -s "$NOTARY_OUT" ]] && { echo "notarytool output:" >&2; cat "$NOTARY_OUT" >&2; echo >&2; }
        if [[ -n "$SUB_ID" ]]; then
            echo "Notarization log for submission $SUB_ID:" >&2
            xcrun notarytool log "$SUB_ID" --keychain-profile "$NOTARY_PROFILE" >&2 \
                || echo "(could not fetch the notarization log)" >&2
        fi
        echo "$OUT is NOT notarized; do not publish it." >&2
        exit 1
    fi
    echo "Notarization accepted (submission $SUB_ID)."

    # 5. Staple the ticket so Gatekeeper can check it offline, then check the result as Gatekeeper does.
    xcrun stapler staple "$OUT" || die "stapling the notarization ticket to $OUT failed"
    xcrun stapler validate "$OUT" || die "the stapled ticket on $OUT does not validate"
    spctl --assess --type install -vv "$OUT" || die "Gatekeeper (spctl) does not accept $OUT"
    NOTARIZED=1
fi

if [[ -n "$PKG_SIGNER" ]]; then
    echo "Package signed with: $PKG_KIND identity"
else
    echo "Note: no \"Developer ID Installer\" identity was used, so the package is UNSIGNED."
fi
echo "Bundle signed with:  ${APP_KIND:-ad-hoc signature}"
if [[ $NOTARIZED == 1 ]]; then
    echo "Notarized and stapled: yes"
else
    echo "Notarized and stapled: no"
    echo "Note: Gatekeeper will block a double-click; see the README for how to open it."
fi
echo "Built and verified: $OUT"
