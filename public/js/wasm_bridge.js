/**
 * MP4-to-ASL — WebAssembly Bridge Module
 * Interface for C++ AnimationEngine WebAssembly binary (bone Slerp and IK solvers).
 * Includes fallback JavaScript vector/quaternion math if WASM is initializing or unavailable.
 */

class WASMBridge {
    constructor() {
        this.wasmLoaded = false;
        this.module = null;
        this.enginePtr = null;
        this._init();
    }

    async _init() {
        if (typeof window.AnimationEngine === 'function') {
            try {
                this.module = await window.AnimationEngine();
                if (this.module && typeof this.module._create_animation_engine === 'function') {
                    this.enginePtr = this.module._create_animation_engine();
                    this.wasmLoaded = true;
                    console.log("[WASMBridge] C++ Animation Engine WebAssembly loaded successfully!");
                }
            } catch (err) {
                console.warn("[WASMBridge] WASM initialization failed, using JS math engine:", err);
            }
        } else {
            console.log("[WASMBridge] Running with JavaScript high-performance math engine.");
        }
    }

    /**
     * Spherical Linear Interpolation (Slerp) between two quaternions [x, y, z, w].
     */
    slerp(q1, q2, t) {
        if (this.wasmLoaded && this.module._slerp_quaternion_wasm) {
            const outPtr = this.module._malloc(16); // 4 floats
            this.module._slerp_quaternion_wasm(
                q1[0], q1[1], q1[2], q1[3],
                q2[0], q2[1], q2[2], q2[3],
                t, outPtr
            );
            const res = [
                this.module.getValue(outPtr, 'float'),
                this.module.getValue(outPtr + 4, 'float'),
                this.module.getValue(outPtr + 8, 'float'),
                this.module.getValue(outPtr + 12, 'float')
            ];
            this.module._free(outPtr);
            return res;
        }

        // Fallback JS Slerp
        return this._jsSlerp(q1, q2, t);
    }

    _jsSlerp(q1, q2, t) {
        let dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];

        if (dot < 0) {
            q2 = [-q2[0], -q2[1], -q2[2], -q2[3]];
            dot = -dot;
        }

        if (dot > 0.9995) {
            const res = [
                q1[0] + t * (q2[0] - q1[0]),
                q1[1] + t * (q2[1] - q1[1]),
                q1[2] + t * (q2[2] - q1[2]),
                q1[3] + t * (q2[3] - q1[3])
            ];
            const len = Math.hypot(...res);
            return res.map(v => v / (len || 1));
        }

        const theta0 = Math.acos(Math.min(1, Math.max(-1, dot)));
        const theta = theta0 * t;
        const sinTheta = Math.sin(theta);
        const sinTheta0 = Math.sin(theta0);

        const s1 = Math.cos(theta) - dot * sinTheta / sinTheta0;
        const s2 = sinTheta / sinTheta0;

        return [
            q1[0] * s1 + q2[0] * s2,
            q1[1] * s1 + q2[1] * s2,
            q1[2] * s1 + q2[2] * s2,
            q1[3] * s1 + q2[3] * s2
        ];
    }
}

window.WASMBridge = WASMBridge;
