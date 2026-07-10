#!/usr/bin/env bash
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$HOME/Qt/6.7.2/gcc_64/lib:$LD_LIBRARY_PATH"
export LIBGL_ALWAYS_SOFTWARE=1   # WSLg hardware GL (zink/d3d12) fails on this host; force software GL
exec ./build/app/cadapp "$@"
