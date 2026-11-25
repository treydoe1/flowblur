#!/bin/bash
# Uninstalls FlowBlur: removes the installed plug-in bundle and forgets the installer receipt.
# Run it as your normal user; sudo asks for your password. Quit Premiere Pro and After Effects first.
set -euo pipefail

TARGET="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/FlowBlur.plugin"
PKG_ID="com.flowblur.motionblur.pkg"

# Refuse to continue unless the path is exactly the fixed MediaCore bundle path.
if [[ "$TARGET" != "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/FlowBlur.plugin" ]]; then
    echo "error: refusing to remove unexpected path: $TARGET" >&2
    exit 1
fi

if [[ -e "$TARGET" || -L "$TARGET" ]]; then
    if [[ ! -L "$TARGET" ]]; then
        ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$TARGET/Contents/Info.plist" 2>/dev/null || true)"
        if [[ "$ID" != "com.flowblur.motionblur" ]]; then
            echo "error: $TARGET is not a FlowBlur bundle (identifier '$ID'); not removing it" >&2
            exit 1
        fi
    fi
    echo "Removing $TARGET"
    sudo /bin/rm -rf "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/FlowBlur.plugin"
else
    echo "FlowBlur.plugin is not installed."
fi

if pkgutil --pkg-info "$PKG_ID" >/dev/null 2>&1; then
    sudo pkgutil --forget "$PKG_ID"
else
    echo "No installer receipt for $PKG_ID."
fi
echo "FlowBlur uninstalled. Restart Premiere Pro / After Effects if they were running."
