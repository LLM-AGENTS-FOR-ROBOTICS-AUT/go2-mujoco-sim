#!/usr/bin/env bash
# Sourced by setup and launch so both enforce the same native Mac requirements.
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup is for macOS. See windows/ or scripts/setup-ubuntu.sh." >&2
  return 1
fi
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -lt 11 ]]; then
  echo "macOS 11 (Big Sur) or newer is required." >&2
  return 1
fi
if [[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || true)" == "1" ]]; then
  echo "Open Terminal without 'Open using Rosetta', then retry for native Apple Silicon." >&2
  return 1
fi

GO2_MAC_ARCH="$(uname -m)"
case "$GO2_MAC_ARCH" in
  arm64) ;;
  x86_64)
    # MuJoCo's Intel binaries require AVX; fail clearly before loading them.
    case " $(sysctl -n machdep.cpu.features 2>/dev/null || true) " in
      *" AVX1.0 "*|*" AVX "*) ;;
      *) echo "MuJoCo requires an Intel Mac with AVX CPU support." >&2; return 1 ;;
    esac
    ;;
  *) echo "Supported Macs: Apple Silicon (arm64) and Intel (x86_64)." >&2; return 1 ;;
esac
