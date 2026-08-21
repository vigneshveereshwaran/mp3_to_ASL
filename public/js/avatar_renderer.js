/**
 * MP4-to-ASL — 3D Avatar Renderer (Three.js WebGL Engine)
 * Renders an articulated humanoid SkinnedMesh & Skeleton bone hierarchy including
 * fully articulated 5-finger hands (Thumb, Index, Middle, Ring, Pinky with 3 joints each).
 */

class AvatarRenderer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.skeleton = null;
        this.animController = null;
        this.clock = new THREE.Clock();
        this._initScene();
    }

    _initScene() {
        const parent = this.canvas.parentElement;
        const width = parent.clientWidth || 600;
        const height = parent.clientHeight || 500;

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.1;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x080b18);
        this.scene.fog = new THREE.FogExp2(0x080b18, 0.08);

        // Camera
        this.camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 50);
        this.camera.position.set(0, 0.25, 2.4);

        // Orbit Controls
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);

        // Lighting
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));

        const keyLight = new THREE.DirectionalLight(0x00f2fe, 2.2);
        keyLight.position.set(3, 5, 4);
        keyLight.castShadow = true;
        this.scene.add(keyLight);

        const rimLight = new THREE.DirectionalLight(0x7f00ff, 2.0);
        rimLight.position.set(-3, 2, -3);
        this.scene.add(rimLight);

        const fillLight = new THREE.PointLight(0x4facfe, 1.2, 8);
        fillLight.position.set(0, 0, 2.2);
        this.scene.add(fillLight);

        // Floor Grid
        const grid = new THREE.GridHelper(12, 36, 0x00f2fe, 0x111827);
        grid.position.y = -1.18;
        this.scene.add(grid);

        // Build Rigged Humanoid Skeleton & Mesh
        this._buildRiggedHumanoid();

        // Initialize Animation Controller
        if (window.AnimationController && this.skeleton) {
            this.animController = new window.AnimationController(this.skeleton);
        }

        // Handle Resize
        window.addEventListener('resize', () => {
            const w = parent.clientWidth;
            const h = parent.clientHeight;
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(w, h);
        });

        // Start Render Loop
        this._renderLoop();
    }

    _mat(color, emissive = 0x000000, emissiveIntensity = 0.0, metalness = 0.6, roughness = 0.3) {
        return new THREE.MeshStandardMaterial({
            color, emissive, emissiveIntensity, metalness, roughness, side: THREE.DoubleSide
        });
    }

    _buildRiggedHumanoid() {
        this.avatarGroup = new THREE.Group();
        const allBones = [];

        // 1. Core Spine / Head Bones
        const hips = new THREE.Bone(); hips.name = "Hips"; hips.position.set(0, -0.4, 0);
        const spine = new THREE.Bone(); spine.name = "Spine"; spine.position.set(0, 0.25, 0);
        const chest = new THREE.Bone(); chest.name = "Chest"; chest.position.set(0, 0.25, 0);
        const neck = new THREE.Bone(); neck.name = "Neck"; neck.position.set(0, 0.25, 0);
        const head = new THREE.Bone(); head.name = "Head"; head.position.set(0, 0.15, 0);

        hips.add(spine); spine.add(chest); chest.add(neck); neck.add(head);
        allBones.push(hips, spine, chest, neck, head);

        // 2. Right Arm & 5 Articulated Fingers
        const rShoulder = new THREE.Bone(); rShoulder.name = "RightShoulder"; rShoulder.position.set(0.16, 0.15, 0);
        const rUpperArm = new THREE.Bone(); rUpperArm.name = "RightUpperArm"; rUpperArm.position.set(0.12, 0, 0);
        const rLowerArm = new THREE.Bone(); rLowerArm.name = "RightLowerArm"; rLowerArm.position.set(0.22, 0, 0);
        const rHand = new THREE.Bone();     rHand.name = "RightHand";     rHand.position.set(0.20, 0, 0);

        chest.add(rShoulder); rShoulder.add(rUpperArm); rUpperArm.add(rLowerArm); rLowerArm.add(rHand);
        allBones.push(rShoulder, rUpperArm, rLowerArm, rHand);

        const fingerConfigs = [
            { name: "Thumb",  y: 0.02,  z: 0.02, len: 0.035, angle: 0.4 },
            { name: "Index",  y: 0.035, z: 0.01, len: 0.045, angle: 0.0 },
            { name: "Middle", y: 0.012, z: 0.01, len: 0.048, angle: 0.0 },
            { name: "Ring",   y: -0.01, z: 0.01, len: 0.044, angle: 0.0 },
            { name: "Pinky",  y: -0.03, z: 0.01, len: 0.038, angle: 0.0 }
        ];

        fingerConfigs.forEach(cfg => {
            const f1 = new THREE.Bone(); f1.name = `Right${cfg.name}1`; f1.position.set(0.04, cfg.y, cfg.z);
            const f2 = new THREE.Bone(); f2.name = `Right${cfg.name}2`; f2.position.set(cfg.len * 0.7, 0, 0);
            const f3 = new THREE.Bone(); f3.name = `Right${cfg.name}3`; f3.position.set(cfg.len * 0.5, 0, 0);
            rHand.add(f1); f1.add(f2); f2.add(f3);
            allBones.push(f1, f2, f3);
        });

        // 3. Left Arm & 5 Articulated Fingers
        const lShoulder = new THREE.Bone(); lShoulder.name = "LeftShoulder"; lShoulder.position.set(-0.16, 0.15, 0);
        const lUpperArm = new THREE.Bone(); lUpperArm.name = "LeftUpperArm"; lUpperArm.position.set(-0.12, 0, 0);
        const lLowerArm = new THREE.Bone(); lLowerArm.name = "LeftLowerArm"; lLowerArm.position.set(-0.22, 0, 0);
        const lHand = new THREE.Bone();     lHand.name = "LeftHand";     lHand.position.set(-0.20, 0, 0);

        chest.add(lShoulder); lShoulder.add(lUpperArm); lUpperArm.add(lLowerArm); lLowerArm.add(lHand);
        allBones.push(lShoulder, lUpperArm, lLowerArm, lHand);

        fingerConfigs.forEach(cfg => {
            const f1 = new THREE.Bone(); f1.name = `Left${cfg.name}1`; f1.position.set(-0.04, cfg.y, cfg.z);
            const f2 = new THREE.Bone(); f2.name = `Left${cfg.name}2`; f2.position.set(-cfg.len * 0.7, 0, 0);
            const f3 = new THREE.Bone(); f3.name = `Left${cfg.name}3`; f3.position.set(-cfg.len * 0.5, 0, 0);
            lHand.add(f1); f1.add(f2); f2.add(f3);
            allBones.push(f1, f2, f3);
        });

        this.skeleton = new THREE.Skeleton(allBones);

        // Materials
        const bodyMat  = this._mat(0x1a2540, 0x050d1e, 0.2, 0.7, 0.3);
        const cyanMat  = this._mat(0x00c8d4, 0x004455, 0.8, 0.9, 0.1);
        const purpMat  = this._mat(0x7f00ff, 0x3a0077, 0.7, 0.8, 0.2);
        const fingerMat = this._mat(0x00f2fe, 0x004455, 0.9, 0.9, 0.1);

        // 4. Attach Meshes to Skeleton Bones
        // Head
        const headGeo = new THREE.SphereGeometry(0.14, 32, 24);
        headGeo.scale(1, 1.18, 0.95);
        const headMesh = new THREE.Mesh(headGeo, bodyMat);
        headMesh.position.set(0, 0.08, 0);
        headMesh.castShadow = true;
        head.add(headMesh);

        const visor = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.032, 0.04), new THREE.MeshBasicMaterial({ color: 0x00f2fe }));
        visor.position.set(0, 0.025, 0.13);
        head.add(visor);

        // Torso
        const torsoMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.14, 0.42, 20), bodyMat);
        torsoMesh.position.set(0, -0.10, 0);
        chest.add(torsoMesh);

        const chestStrip = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.32, 0.015), cyanMat);
        chestStrip.position.set(0, -0.05, 0.18);
        chest.add(chestStrip);

        // Right Arm & Finger Meshes
        const rUpperMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.036, 0.22, 14), cyanMat);
        rUpperMesh.rotation.z = -Math.PI / 2; rUpperMesh.position.set(0.11, 0, 0);
        rUpperArm.add(rUpperMesh);

        const rLowerMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.030, 0.20, 14), purpMat);
        rLowerMesh.rotation.z = -Math.PI / 2; rLowerMesh.position.set(0.10, 0, 0);
        rLowerArm.add(rLowerMesh);

        const rPalmMesh = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.075, 0.025), cyanMat);
        rPalmMesh.position.set(0.03, 0, 0);
        rHand.add(rPalmMesh);

        // Right Fingers
        fingerConfigs.forEach(cfg => {
            const f1Bone = rHand.getObjectByName(`Right${cfg.name}1`);
            const f2Bone = rHand.getObjectByName(`Right${cfg.name}2`);
            const f3Bone = rHand.getObjectByName(`Right${cfg.name}3`);

            if (f1Bone && f2Bone && f3Bone) {
                const seg1 = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.008, cfg.len * 0.7, 8), fingerMat);
                seg1.rotation.z = -Math.PI / 2; seg1.position.set(cfg.len * 0.35, 0, 0);
                f1Bone.add(seg1);

                const seg2 = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.007, cfg.len * 0.5, 8), fingerMat);
                seg2.rotation.z = -Math.PI / 2; seg2.position.set(cfg.len * 0.25, 0, 0);
                f2Bone.add(seg2);

                const seg3 = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.005, cfg.len * 0.4, 8), fingerMat);
                seg3.rotation.z = -Math.PI / 2; seg3.position.set(cfg.len * 0.2, 0, 0);
                f3Bone.add(seg3);
            }
        });

        // Left Arm & Finger Meshes
        const lUpperMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.036, 0.22, 14), cyanMat);
        lUpperMesh.rotation.z = Math.PI / 2; lUpperMesh.position.set(-0.11, 0, 0);
        lUpperArm.add(lUpperMesh);

        const lLowerMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.030, 0.20, 14), purpMat);
        lLowerMesh.rotation.z = Math.PI / 2; lLowerMesh.position.set(-0.10, 0, 0);
        lLowerArm.add(lLowerMesh);

        const lPalmMesh = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.075, 0.025), cyanMat);
        lPalmMesh.position.set(-0.03, 0, 0);
        lHand.add(lPalmMesh);

        // Left Fingers
        fingerConfigs.forEach(cfg => {
            const f1Bone = lHand.getObjectByName(`Left${cfg.name}1`);
            const f2Bone = lHand.getObjectByName(`Left${cfg.name}2`);
            const f3Bone = lHand.getObjectByName(`Left${cfg.name}3`);

            if (f1Bone && f2Bone && f3Bone) {
                const seg1 = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.008, cfg.len * 0.7, 8), fingerMat);
                seg1.rotation.z = Math.PI / 2; seg1.position.set(-cfg.len * 0.35, 0, 0);
                f1Bone.add(seg1);

                const seg2 = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.007, cfg.len * 0.5, 8), fingerMat);
                seg2.rotation.z = Math.PI / 2; seg2.position.set(-cfg.len * 0.25, 0, 0);
                f2Bone.add(seg2);

                const seg3 = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.005, cfg.len * 0.4, 8), fingerMat);
                seg3.rotation.z = Math.PI / 2; seg3.position.set(-cfg.len * 0.2, 0, 0);
                f3Bone.add(seg3);
            }
        });

        // Waist / Legs
        const waist = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.115, 0.12, 16), purpMat);
        waist.position.set(0, -0.15, 0);
        hips.add(waist);

        [-0.07, 0.07].forEach(x => {
            const thigh = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.055, 0.28, 14), bodyMat);
            thigh.position.set(x, -0.38, 0);
            hips.add(thigh);

            const shin = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.038, 0.27, 14), bodyMat);
            shin.position.set(x, -0.65, 0.02);
            hips.add(shin);
        });

        this.avatarGroup.add(hips);
        this.scene.add(this.avatarGroup);
    }

    playSignSequence(glossTokens, onSignChange = null) {
        if (this.animController) {
            this.animController.playSignSequence(glossTokens, onSignChange);
        }
    }

    _renderLoop() {
        requestAnimationFrame(() => this._renderLoop());

        const delta = this.clock.getDelta();
        if (this.animController) {
            this.animController.update(delta);
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

window.AvatarRenderer = AvatarRenderer;
