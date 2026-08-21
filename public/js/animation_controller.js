/**
 * MP4-to-ASL — Animation Controller & ASL Keyframe Pose Library
 * Manages skeletal bone rotations, finger joint handshapes, Spherical Linear Interpolation (Slerp),
 * sequence queuing, and smooth transitions between distinct ASL signs.
 */

class AnimationController {
    constructor(skeleton) {
        this.skeleton = skeleton;
        this.boneMap = {};
        this._mapBones();

        this.currentPose = {};
        this.targetPose = {};
        this.startPose = {};

        this.isPlaying = false;
        this.signQueue = [];
        this.currentSignIndex = 0;
        this.keyframeIndex = 0;

        this.transitionDuration = 0.25; // 250ms transition
        this.keyframeDuration = 0.65;   // 650ms per keyframe
        this.elapsedTime = 0.0;
        this.signCallback = null;

        // Helper for degrees -> radians
        this.deg = (degrees) => degrees * (Math.PI / 180.0);

        // Load Pose Definitions
        this.poseLibrary = this._buildPoseLibrary();

        // Set Neutral Resting Stance
        this.setNeutralPose();
    }

    _mapBones() {
        if (!this.skeleton || !this.skeleton.bones) return;
        this.skeleton.bones.forEach(bone => {
            this.boneMap[bone.name] = bone;
        });
    }

    setNeutralPose() {
        const neutral = this.poseLibrary["NEUTRAL"] || {};
        Object.keys(neutral).forEach(boneName => {
            if (this.boneMap[boneName]) {
                const rot = neutral[boneName];
                this.boneMap[boneName].rotation.set(rot.x || 0, rot.y || 0, rot.z || 0);
            }
        });
    }

    // Handshape presets for 5 fingers (3 joints each)
    _handFist(prefix = "Right") {
        const d = this.deg;
        const res = {};
        ["Index", "Middle", "Ring", "Pinky"].forEach(f => {
            res[`${prefix}${f}1`] = { x: d(80), y: d(0), z: d(0) };
            res[`${prefix}${f}2`] = { x: d(90), y: d(0), z: d(0) };
            res[`${prefix}${f}3`] = { x: d(70), y: d(0), z: d(0) };
        });
        res[`${prefix}Thumb1`] = { x: d(20), y: d(40), z: d(20) };
        res[`${prefix}Thumb2`] = { x: d(40), y: d(0), z: d(0) };
        res[`${prefix}Thumb3`] = { x: d(40), y: d(0), z: d(0) };
        return res;
    }

    _handFlat(prefix = "Right") {
        const d = this.deg;
        const res = {};
        ["Thumb", "Index", "Middle", "Ring", "Pinky"].forEach(f => {
            res[`${prefix}${f}1`] = { x: d(0), y: d(0), z: d(0) };
            res[`${prefix}${f}2`] = { x: d(0), y: d(0), z: d(0) };
            res[`${prefix}${f}3`] = { x: d(0), y: d(0), z: d(0) };
        });
        return res;
    }

    _handPointIndex(prefix = "Right") {
        const res = this._handFist(prefix);
        const d = this.deg;
        // Extend Index straight out
        res[`${prefix}Index1`] = { x: d(0), y: d(0), z: d(0) };
        res[`${prefix}Index2`] = { x: d(0), y: d(0), z: d(0) };
        res[`${prefix}Index3`] = { x: d(0), y: d(0), z: d(0) };
        return res;
    }

    _handPeaceV(prefix = "Right") {
        const res = this._handFist(prefix);
        const d = this.deg;
        // Extend Index and Middle
        res[`${prefix}Index1`] = { x: d(0), y: d(0), z: d(-10) };
        res[`${prefix}Index2`] = { x: d(0), y: d(0), z: d(0) };
        res[`${prefix}Index3`] = { x: d(0), y: d(0), z: d(0) };
        res[`${prefix}Middle1`] = { x: d(0), y: d(0), z: d(10) };
        res[`${prefix}Middle2`] = { x: d(0), y: d(0), z: d(0) };
        res[`${prefix}Middle3`] = { x: d(0), y: d(0), z: d(0) };
        return res;
    }

    _handClaw(prefix = "Right") {
        const d = this.deg;
        const res = {};
        ["Thumb", "Index", "Middle", "Ring", "Pinky"].forEach(f => {
            res[`${prefix}${f}1`] = { x: d(45), y: d(0), z: d(0) };
            res[`${prefix}${f}2`] = { x: d(50), y: d(0), z: d(0) };
            res[`${prefix}${f}3`] = { x: d(40), y: d(0), z: d(0) };
        });
        return res;
    }

    _handOKSign(prefix = "Right") {
        const res = this._handFlat(prefix);
        const d = this.deg;
        // Touch Index tip and Thumb tip
        res[`${prefix}Index1`] = { x: d(60), y: d(10), z: d(0) };
        res[`${prefix}Index2`] = { x: d(60), y: d(0), z: d(0) };
        res[`${prefix}Thumb1`] = { x: d(45), y: d(30), z: d(0) };
        res[`${prefix}Thumb2`] = { x: d(45), y: d(0), z: d(0) };
        return res;
    }

    _buildPoseLibrary() {
        const d = this.deg;
        const deg = this.deg;

        return {
            "NEUTRAL": Object.assign({
                "RightShoulder": { x: d(0), y: d(0), z: d(-10) },
                "RightUpperArm": { x: d(10), y: d(0), z: d(-75) },
                "RightLowerArm": { x: d(20), y: d(0), z: d(35) },
                "RightHand":     { x: d(0), y: d(0), z: d(0) },
                "LeftShoulder":  { x: d(0), y: d(0), z: d(10) },
                "LeftUpperArm":  { x: d(10), y: d(0), z: d(75) },
                "LeftLowerArm":  { x: d(20), y: d(0), z: d(-35) },
                "LeftHand":      { x: d(0), y: d(0), z: d(0) },
                "Head":          { x: d(0), y: d(0), z: d(0) },
                "Chest":         { x: d(0), y: d(0), z: d(0) }
            }, this._handFlat("Right"), this._handFlat("Left")),

            // HELLO: Open hand wave near head (Flat open handshape)
            "HELLO": [
                Object.assign({
                    "RightShoulder": { x: d(15), y: d(20), z: d(10) },
                    "RightUpperArm": { x: d(-20), y: d(30), z: deg(75) },
                    "RightLowerArm": { x: d(95), y: d(20), z: d(45) },
                    "RightHand":     { x: d(0), y: d(15), z: d(-20) },
                    "Head":          { x: d(-5), y: d(10), z: d(0) }
                }, this._handFlat("Right")),
                Object.assign({
                    "RightShoulder": { x: d(15), y: d(20), z: d(10) },
                    "RightUpperArm": { x: d(-15), y: d(35), z: d(80) },
                    "RightLowerArm": { x: d(105), y: d(30), z: d(60) },
                    "RightHand":     { x: d(0), y: d(-15), z: d(20) },
                    "Head":          { x: d(0), y: d(0), z: d(0) }
                }, this._handFlat("Right"))
            ],

            // GOOD: Open B-hand touches chin, then moves down onto flat left palm
            "GOOD": [
                Object.assign({
                    "RightUpperArm": { x: d(-30), y: d(20), z: d(60) },
                    "RightLowerArm": { x: d(125), y: d(10), z: d(30) },
                    "RightHand":     { x: d(15), y: d(0), z: d(0) },
                    "Head":          { x: d(10), y: d(0), z: d(0) }
                }, this._handFlat("Right")),
                Object.assign({
                    "RightUpperArm": { x: d(15), y: d(10), z: d(35) },
                    "RightLowerArm": { x: d(60), y: d(0), z: d(10) },
                    "RightHand":     { x: d(-10), y: deg(0), z: d(0) },
                    "LeftUpperArm":  { x: d(15), y: d(-10), z: d(-35) },
                    "LeftLowerArm":  { x: d(70), y: d(0), z: d(-10) },
                    "LeftHand":      { x: d(0), y: d(0), z: d(0) }
                }, this._handFlat("Right"), this._handFlat("Left"))
            ],

            // MORNING: Left arm horizontal (horizon), right B-hand rises up like the sun
            "MORNING": [
                Object.assign({
                    "LeftUpperArm":  { x: d(0), y: d(-30), z: d(-65) },
                    "LeftLowerArm":  { x: d(85), y: d(0), z: d(0) },
                    "RightUpperArm": { x: d(30), y: d(10), z: d(20) },
                    "RightLowerArm": { x: d(40), y: d(0), z: d(10) }
                }, this._handFlat("Left"), this._handFlat("Right")),
                Object.assign({
                    "LeftUpperArm":  { x: d(0), y: d(-30), z: d(-65) },
                    "LeftLowerArm":  { x: d(85), y: d(0), z: d(0) },
                    "RightUpperArm": { x: d(-25), y: d(20), z: d(55) },
                    "RightLowerArm": { x: d(110), y: d(10), z: d(25) },
                    "RightHand":     { x: d(10), y: d(0), z: d(0) }
                }, this._handFlat("Left"), this._handFlat("Right"))
            ],

            // THANK-YOU: Open flat hand from chin outward
            "THANK-YOU": [
                Object.assign({
                    "RightUpperArm": { x: d(-25), y: d(15), z: d(55) },
                    "RightLowerArm": { x: d(120), y: d(0), z: d(20) },
                    "RightHand":     { x: d(10), y: d(0), z: d(0) }
                }, this._handFlat("Right")),
                Object.assign({
                    "RightUpperArm": { x: d(10), y: d(30), z: d(45) },
                    "RightLowerArm": { x: d(45), y: d(0), z: d(10) },
                    "RightHand":     { x: d(-15), y: d(0), z: d(0) }
                }, this._handFlat("Right"))
            ],

            // YES: Fist (S handshape) nodding up and down
            "YES": [
                Object.assign({
                    "RightUpperArm": { x: d(0), y: d(20), z: d(45) },
                    "RightLowerArm": { x: d(90), y: d(0), z: d(20) },
                    "RightHand":     { x: d(35), y: d(0), z: d(0) }
                }, this._handFist("Right")),
                Object.assign({
                    "RightUpperArm": { x: d(0), y: d(20), z: d(45) },
                    "RightLowerArm": { x: d(75), y: d(0), z: d(20) },
                    "RightHand":     { x: d(-20), y: d(0), z: d(0) }
                }, this._handFist("Right"))
            ],

            // NO: Index + middle finger snap down onto thumb
            "NO": [
                Object.assign({
                    "RightUpperArm": { x: d(-10), y: d(25), z: d(50) },
                    "RightLowerArm": { x: d(95), y: d(10), z: d(15) },
                    "RightHand":     { x: d(-15), y: d(20), z: d(0) }
                }, this._handPeaceV("Right")),
                Object.assign({
                    "RightUpperArm": { x: d(-10), y: d(25), z: d(50) },
                    "RightLowerArm": { x: d(85), y: d(10), z: d(15) },
                    "RightHand":     { x: d(25), y: d(-10), z: d(0) }
                }, this._handFist("Right"))
            ],

            // I / IX-1: Index finger points directly to chest
            "IX-1": [
                Object.assign({
                    "RightUpperArm": { x: d(10), y: d(-15), z: d(35) },
                    "RightLowerArm": { x: d(105), y: d(-20), z: d(25) },
                    "RightHand":     { x: d(20), y: d(-30), z: d(0) }
                }, this._handPointIndex("Right"))
            ],
            "I": [
                Object.assign({
                    "RightUpperArm": { x: d(10), y: d(-15), z: d(35) },
                    "RightLowerArm": { x: d(105), y: d(-20), z: d(25) },
                    "RightHand":     { x: d(20), y: d(-30), z: d(0) }
                }, this._handPointIndex("Right"))
            ],

            // YOU / IX-2: Index finger points forward
            "IX-2": [
                Object.assign({
                    "RightUpperArm": { x: d(0), y: d(30), z: d(50) },
                    "RightLowerArm": { x: d(50), y: d(10), z: d(10) },
                    "RightHand":     { x: d(-10), y: d(0), z: d(0) }
                }, this._handPointIndex("Right"))
            ],
            "YOU": [
                Object.assign({
                    "RightUpperArm": { x: d(0), y: d(30), z: d(50) },
                    "RightLowerArm": { x: d(50), y: d(10), z: d(10) },
                    "RightHand":     { x: d(-10), y: d(0), z: d(0) }
                }, this._handPointIndex("Right"))
            ],

            // NAME: Double tap of H-handshapes (Index+Middle straight)
            "NAME": [
                Object.assign({
                    "RightUpperArm": { x: d(10), y: d(15), z: d(40) },
                    "RightLowerArm": { x: d(85), y: d(0), z: d(15) },
                    "LeftUpperArm":  { x: d(10), y: d(-15), z: d(-40) },
                    "LeftLowerArm":  { x: d(85), y: d(0), z: d(-15) }
                }, this._handPeaceV("Right"), this._handPeaceV("Left")),
                Object.assign({
                    "RightUpperArm": { x: d(10), y: d(15), z: d(40) },
                    "RightLowerArm": { x: d(75), y: d(0), z: d(15) },
                    "LeftUpperArm":  { x: d(10), y: d(-15), z: d(-40) },
                    "LeftLowerArm":  { x: d(85), y: d(0), z: d(-15) }
                }, this._handPeaceV("Right"), this._handPeaceV("Left"))
            ],

            // WANT: Both hands pull back with Claw handshapes
            "WANT": [
                Object.assign({
                    "RightUpperArm": { x: d(10), y: d(20), z: d(45) },
                    "RightLowerArm": { x: d(75), y: d(10), z: d(10) },
                    "LeftUpperArm":  { x: d(10), y: d(-20), z: d(-45) },
                    "LeftLowerArm":  { x: d(75), y: d(-10), z: d(-10) }
                }, this._handClaw("Right"), this._handClaw("Left")),
                Object.assign({
                    "RightUpperArm": { x: d(25), y: d(25), z: d(50) },
                    "RightLowerArm": { x: d(50), y: d(10), z: d(10) },
                    "LeftUpperArm":  { x: d(25), y: d(-25), z: d(-50) },
                    "LeftLowerArm":  { x: d(50), y: d(-10), z: d(-10) }
                }, this._handClaw("Right"), this._handClaw("Left"))
            ],

            // WHERE: Index finger shakes side to side (PointIndex handshape)
            "WHERE": [
                Object.assign({
                    "RightUpperArm": { x: d(-10), y: d(25), z: d(55) },
                    "RightLowerArm": { x: d(100), y: d(10), z: d(20) },
                    "RightHand":     { x: d(0), y: d(25), z: d(0) }
                }, this._handPointIndex("Right")),
                Object.assign({
                    "RightUpperArm": { x: d(-10), y: d(25), z: d(55) },
                    "RightLowerArm": { x: d(100), y: d(10), z: d(20) },
                    "RightHand":     { x: d(0), y: d(-25), z: d(0) }
                }, this._handPointIndex("Right"))
            ],

            // WHAT: Both open palms facing up, shaking side to side
            "WHAT": [
                Object.assign({
                    "RightUpperArm": { x: d(10), y: d(35), z: d(45) },
                    "RightLowerArm": { x: d(70), y: d(20), z: d(30) },
                    "LeftUpperArm":  { x: d(10), y: d(-35), z: d(-45) },
                    "LeftLowerArm":  { x: d(70), y: d(-20), z: d(-30) }
                }, this._handFlat("Right"), this._handFlat("Left")),
                Object.assign({
                    "RightUpperArm": { x: d(15), y: d(40), z: d(50) },
                    "RightLowerArm": { x: d(60), y: d(25), z: d(35) },
                    "LeftUpperArm":  { x: d(15), y: d(-40), z: d(-50) },
                    "LeftLowerArm":  { x: d(60), y: d(-25), z: d(-30) }
                }, this._handFlat("Right"), this._handFlat("Left"))
            ],

            // YESTERDAY: A/Thumb handshape touches cheek, moves back to ear
            "YESTERDAY": [
                Object.assign({
                    "RightUpperArm": { x: d(-20), y: d(15), z: d(65) },
                    "RightLowerArm": { x: d(125), y: d(10), z: d(25) },
                    "RightHand":     { x: d(15), y: d(10), z: d(0) }
                }, this._handFist("Right")),
                Object.assign({
                    "RightUpperArm": { x: d(-30), y: d(10), z: d(70) },
                    "RightLowerArm": { x: d(135), y: d(5), z: d(30) },
                    "RightHand":     { x: d(25), y: d(15), z: d(0) }
                }, this._handFist("Right"))
            ],

            // SCHOOL: Flat hands clap horizontally twice
            "SCHOOL": [
                Object.assign({
                    "RightUpperArm": { x: d(15), y: d(15), z: d(40) },
                    "RightLowerArm": { x: d(85), y: d(0), z: d(20) },
                    "LeftUpperArm":  { x: d(15), y: d(-15), z: d(-40) },
                    "LeftLowerArm":  { x: d(85), y: d(0), z: d(-20) }
                }, this._handFlat("Right"), this._handFlat("Left")),
                Object.assign({
                    "RightUpperArm": { x: d(20), y: d(15), z: d(45) },
                    "RightLowerArm": { x: d(65), y: d(0), z: d(20) },
                    "LeftUpperArm":  { x: d(15), y: d(-15), z: d(-40) },
                    "LeftLowerArm":  { x: d(85), y: d(0), z: d(-20) }
                }, this._handFlat("Right"), this._handFlat("Left"))
            ]
        };
    }

    playSignSequence(glossTokens, onSignChange = null) {
        if (!glossTokens || glossTokens.length === 0) return;

        this.signQueue = [...glossTokens];
        this.currentSignIndex = 0;
        this.keyframeIndex = 0;
        this.isPlaying = true;
        this.elapsedTime = 0.0;
        this.signCallback = onSignChange;

        this._startNextSign();
    }

    _startNextSign() {
        if (this.currentSignIndex >= this.signQueue.length) {
            this.isPlaying = false;
            this.setNeutralPose();
            if (this.signCallback) this.signCallback(null);
            return;
        }

        const currentToken = this.signQueue[this.currentSignIndex].toUpperCase();
        if (this.signCallback) this.signCallback(currentToken);

        let signKeyframes = this.poseLibrary[currentToken];
        if (!signKeyframes) {
            if (currentToken.startsWith("IX-")) {
                signKeyframes = this.poseLibrary["IX-1"];
            } else {
                signKeyframes = this.poseLibrary["HELLO"];
            }
        }

        if (!Array.isArray(signKeyframes)) {
            signKeyframes = [signKeyframes];
        }

        this.currentSignKeyframes = signKeyframes;
        this.keyframeIndex = 0;

        this._captureCurrentPoseAsStart();
        this._setTargetKeyframe(this.currentSignKeyframes[0]);
    }

    _captureCurrentPoseAsStart() {
        this.startPose = {};
        Object.keys(this.boneMap).forEach(boneName => {
            const bone = this.boneMap[boneName];
            this.startPose[boneName] = {
                x: bone.rotation.x,
                y: bone.rotation.y,
                z: bone.rotation.z
            };
        });
    }

    _setTargetKeyframe(keyframe) {
        this.targetPose = keyframe;
        this.elapsedTime = 0.0;
    }

    update(deltaTime) {
        if (!this.isPlaying || !this.currentSignKeyframes) return;

        this.elapsedTime += deltaTime;
        const progress = Math.min(1.0, this.elapsedTime / this.keyframeDuration);

        // Smooth cosine easing
        const easeT = 0.5 * (1.0 - Math.cos(progress * Math.PI));

        // Interpolate all active bone rotations (arms, hands, fingers)
        Object.keys(this.targetPose).forEach(boneName => {
            const bone = this.boneMap[boneName];
            if (bone && this.startPose[boneName]) {
                const start = this.startPose[boneName];
                const target = this.targetPose[boneName];

                bone.rotation.x = start.x + (target.x - start.x) * easeT;
                bone.rotation.y = start.y + (target.y - start.y) * easeT;
                bone.rotation.z = start.z + (target.z - start.z) * easeT;
            }
        });

        // Move to next keyframe or next sign
        if (progress >= 1.0) {
            this.keyframeIndex++;
            if (this.keyframeIndex < this.currentSignKeyframes.length) {
                this._captureCurrentPoseAsStart();
                this._setTargetKeyframe(this.currentSignKeyframes[this.keyframeIndex]);
            } else {
                this.currentSignIndex++;
                this._startNextSign();
            }
        }
    }
}

window.AnimationController = AnimationController;
