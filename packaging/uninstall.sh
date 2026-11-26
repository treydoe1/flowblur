#!/bin/bash
# Uninstalls T Blur: removes the installed plug-in bundle and forgets the installer receipt.
# T Blur was called FlowBlur up to version 0.6.0; a FlowBlur install is removed the same way.
# Run it as your normal user; sudo asks for your password. Quit Premiere Pro and After Effects first.
set -euo pipefail

MEDIACORE="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"

# remove_bundle <bundle file name> <expected CFBundleIdentifier> <installer receipt id> <product name>
remove_bundle() {
    local name="$1" want_id="$2" pkg_id="$3" product="$4"
    local target="$MEDIACORE/$name"
    # Refuse to continue unless the path is one of the two fixed MediaCore bundle paths.
    if [[ "$target" != "$MEDIACORE/TBlur.plugin" && "$target" != "$MEDIACORE/FlowBlur.plugin" ]]; then
        echo "error: refusing to remove unexpected path: $target" >&2
        exit 1
    fi
    if [[ -e "$target" || -L "$target" ]]; then
        if [[ ! -L "$target" ]]; then
            local id
            id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$target/Contents/Info.plist" 2>/dev/null || true)"
            if [[ "$id" != "$want_id" ]]; then
                echo "error: $target is not a $product bundle (identifier '$id'); not removing it" >&2
                exit 1
            fi
        fi
        echo "Removing $target"
        sudo /bin/rm -rf "$target"
        FOUND=1
    fi
    if pkgutil --pkg-info "$pkg_id" >/dev/null 2>&1; then
        sudo pkgutil --forget "$pkg_id"
        FOUND=1
    fi
}

FOUND=0
remove_bundle TBlur.plugin com.tblur.motionblur com.tblur.motionblur.pkg "T Blur"
remove_bundle FlowBlur.plugin com.flowblur.motionblur com.flowblur.motionblur.pkg "FlowBlur"

if [[ $FOUND == 1 ]]; then
    echo "T Blur uninstalled. Restart Premiere Pro / After Effects if they were running."
else
    echo "T Blur is not installed."
fi
# The update check keeps a small state folder in each user's home directory. It is not removed
# here: this script may be run under sudo, where the home directory is not reliably the user's.
echo "The update-check folder ~/Library/Application Support/TBlur, if present, is left in place"
echo "(FlowBlur 0.6.0 used ~/Library/Application Support/FlowBlur). To remove them, run as yourself:"
echo "  rm -rf ~/Library/Application\\ Support/TBlur ~/Library/Application\\ Support/FlowBlur"
