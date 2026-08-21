/**
 * HearLink ASL — WASM Bridge
 * Marshals keypoint pose arrays into C++ WebAssembly heap memory, invokes SLERP / IK,
 * and falls back to JavaScript interpolation when WASM is loading/unavailable.
 */

export class WASMAnimationBridge {
    constructor() {
        this.wasmModule = null;
        this.enginePtr = null;
        this.isReady = false;
        this.inputPtr = null;
        this.outputPtr = null;
    }

    async init() {
        try {
            if (window.AnimationEngine) {
                this.wasmModule = await window.AnimationEngine();
                this.enginePtr = this.wasmModule._create_animation_engine();

                // Allocate WASM heap buffers
                // 67 landmarks * 3 floats = 201 floats for input
                // 67 landmarks * 7 floats (quat + pos) = 469 floats for output
                this.inputPtr = this.wasmModule._malloc(201 * 4);
                this.outputPtr = this.wasmModule._malloc(469 * 4);

                this.isReady = true;
                console.log("[WASMBridge] C++ WebAssembly Animation Engine initialized successfully!");
            } else {
                console.warn("[WASMBridge] AnimationEngine JS wrapper not loaded. Using JS fallback.");
            }
        } catch (err) {
            console.warn("[WASMBridge] WASM initialization failed, switching to JS fallback:", err);
            this.isReady = false;
        }
    }

    processFrame(landmarks, dt = 0.033) {
        if (this.isReady && this.wasmModule) {
            return this._processWASM(landmarks, dt);
        } else {
            return this._processJSFallback(landmarks);
        }
    }

    _processWASM(landmarks, dt) {
        // Flatten landmarks array into input heap
        const inputData = new Float32Array(201);
        let idx = 0;

        const pose = landmarks.pose || [];
        for (let i = 0; i < 25; i++) {
            if (i < pose.length) {
                inputData[idx++] = pose[i][0];
                inputData[idx++] = pose[i][1];
                inputData[idx++] = pose[i][2];
            } else {
                inputData[idx++] = 0; inputData[idx++] = 0; inputData[idx++] = 0;
            }
        }

        const rHand = landmarks.right_hand || [];
        for (let i = 0; i < 21; i++) {
            if (i < rHand.length) {
                inputData[idx++] = rHand[i][0];
                inputData[idx++] = rHand[i][1];
                inputData[idx++] = rHand[i][2];
            } else {
                inputData[idx++] = 0; inputData[idx++] = 0; inputData[idx++] = 0;
            }
        }

        const lHand = landmarks.left_hand || [];
        for (let i = 0; i < 21; i++) {
            if (i < lHand.length) {
                inputData[idx++] = lHand[i][0];
                inputData[idx++] = lHand[i][1];
                inputData[idx++] = lHand[i][2];
            } else {
                inputData[idx++] = 0; inputData[idx++] = 0; inputData[idx++] = 0;
            }
        }

        // Copy input to WASM memory
        this.wasmModule.HEAPF32.set(inputData, this.inputPtr >> 2);

        // Call WASM function
        this.wasmModule._process_pose_frame_wasm(this.enginePtr, this.inputPtr, 67, dt, this.outputPtr);

        // Read output from WASM memory
        const outputData = new Float32Array(this.wasmModule.HEAPF32.buffer, this.outputPtr, 469);

        // Format output into joint transform objects
        const transforms = [];
        for (let i = 0; i < 67; i++) {
            const base = i * 7;
            transforms.push({
                rotation: [outputData[base], outputData[base + 1], outputData[base + 2], outputData[base + 3]],
                position: [outputData[base + 4], outputData[base + 5], outputData[base + 6]]
            });
        }

        return transforms;
    }

    _processJSFallback(landmarks) {
        // Pure JS fallback when WASM is disabled or loading
        const transforms = [];
        const pose = landmarks.pose || [];
        for (let i = 0; i < 25; i++) {
            const pos = i < pose.length ? pose[i] : [0, 0, 0];
            transforms.push({
                rotation: [0, 0, 0, 1],
                position: pos
            });
        }
        return transforms;
    }
}
