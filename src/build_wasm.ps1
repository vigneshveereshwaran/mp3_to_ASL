# PowerShell build script for Emscripten C++ -> WebAssembly compilation

Write-Host "Compiling HearLink ASL Animation Engine to WebAssembly..."

New-Item -ItemType Directory -Force -Path "public/wasm" | Out-Null

emcc src/animation_engine.cpp -O3 -s WASM=1 -s EXPORTED_FUNCTIONS="['_create_animation_engine','_destroy_animation_engine','_process_pose_frame_wasm','_slerp_quaternion_wasm','_malloc','_free']" -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','getValue','setValue']" -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME="AnimationEngine" -o public/wasm/animation_engine.js

Write-Host "WebAssembly compilation finished -> public/wasm/animation_engine.js"
