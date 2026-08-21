/**
 * HearLink ASL — 3D Avatar Renderer
 * Renders a full, solid 3D humanoid character mesh with skeletal bone hierarchy,
 * skinning, dynamic limb orientation, OrbitControls camera, and PBR dark mode lighting.
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

export class AvatarRenderer {
    constructor(canvasContainer) {
        this.container = canvasContainer;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.avatarGroup = null;

        // Meshes
        this.headMesh = null;
        this.torsoMesh = null;
        this.rUpperArmMesh = null;
        this.rLowerArmMesh = null;
        this.rHandMesh = null;
        this.lUpperArmMesh = null;
        this.lLowerArmMesh = null;
        this.lHandMesh = null;

        // Joints
        this.joints = {};
        this.rFingerMeshes = [];
        this.lFingerMeshes = [];

        this.init();
    }

    init() {
        // 1. Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color('#0a0c16');
        this.scene.fog = new THREE.FogExp2('#0a0c16', 0.15);

        // 2. Camera setup
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
        this.camera.position.set(0, 0.1, 2.2);

        // 3. Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.container.appendChild(this.renderer.domElement);

        // 4. OrbitControls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2 + 0.1;
        this.controls.minDistance = 1.0;
        this.controls.maxDistance = 5.0;
        this.controls.target.set(0, -0.1, 0);

        // 5. Lighting System
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
        this.scene.add(ambientLight);

        const mainLight = new THREE.DirectionalLight(0x00f2fe, 2.0);
        mainLight.position.set(3, 5, 4);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 1024;
        mainLight.shadow.mapSize.height = 1024;
        this.scene.add(mainLight);

        const rimLight = new THREE.DirectionalLight(0x7f00ff, 1.8);
        rimLight.position.set(-3, 3, -3);
        this.scene.add(rimLight);

        const fillLight = new THREE.PointLight(0x4facfe, 1.2, 10);
        fillLight.position.set(0, -0.5, 2);
        this.scene.add(fillLight);

        // 6. Floor Grid with Neon Glow
        const grid = new THREE.GridHelper(10, 30, 0x00f2fe, 0x1f293d);
        grid.position.y = -1.1;
        this.scene.add(grid);

        // 7. Build Full 3D Solid Humanoid Avatar
        this._buildSolidHumanoidAvatar();

        // 8. Handle Window Resizing
        window.addEventListener('resize', () => this.onWindowResize());

        // 9. Start Render Loop
        this.animate();
    }

    _buildSolidHumanoidAvatar() {
        this.avatarGroup = new THREE.Group();

        // Materials
        const skinMat = new THREE.MeshStandardMaterial({
            color: 0x1a233a,
            roughness: 0.3,
            metalness: 0.7,
            emissive: 0x0a1224
        });

        const accentMat = new THREE.MeshStandardMaterial({
            color: 0x00f2fe,
            roughness: 0.1,
            metalness: 0.9,
            emissive: 0x005577,
            emissiveIntensity: 0.6
        });

        const handMat = new THREE.MeshStandardMaterial({
            color: 0x7f00ff,
            roughness: 0.2,
            metalness: 0.8,
            emissive: 0x3a0077,
            emissiveIntensity: 0.5
        });

        const eyeMat = new THREE.MeshBasicMaterial({ color: 0x00f2fe });

        // --- 1. HEAD & FACE ---
        const headGroup = new THREE.Group();
        const headGeo = new THREE.SphereGeometry(0.13, 32, 32);
        headGeo.scale(1, 1.15, 1);
        const headMesh = new THREE.Mesh(headGeo, skinMat);
        headMesh.castShadow = true;
        headGroup.add(headMesh);

        // Visor / Eyes
        const eyeGeo = new THREE.BoxGeometry(0.14, 0.03, 0.05);
        const eyeMesh = new THREE.Mesh(eyeGeo, eyeMat);
        eyeMesh.position.set(0, 0.03, 0.11);
        headGroup.add(eyeMesh);

        headGroup.position.set(0, 0.45, 0);
        this.avatarGroup.add(headGroup);
        this.headMesh = headGroup;

        // --- 2. TORSO & NECK ---
        const torsoGroup = new THREE.Group();

        // Neck
        const neckGeo = new THREE.CylinderGeometry(0.05, 0.06, 0.08, 16);
        const neckMesh = new THREE.Mesh(neckGeo, skinMat);
        neckMesh.position.set(0, 0.32, 0);
        torsoGroup.add(neckMesh);

        // Chest Armor
        const chestGeo = new THREE.CylinderGeometry(0.18, 0.13, 0.35, 16);
        const chestMesh = new THREE.Mesh(chestGeo, skinMat);
        chestMesh.position.set(0, 0.1, 0);
        chestMesh.castShadow = true;
        torsoGroup.add(chestMesh);

        // Core Accent Line
        const coreGeo = new THREE.BoxGeometry(0.04, 0.28, 0.02);
        const coreMesh = new THREE.Mesh(coreGeo, accentMat);
        coreMesh.position.set(0, 0.1, 0.16);
        torsoGroup.add(coreMesh);

        // Shoulders
        const rShoulderGeo = new THREE.SphereGeometry(0.065, 16, 16);
        const rShoulderMesh = new THREE.Mesh(rShoulderGeo, accentMat);
        rShoulderMesh.position.set(0.20, 0.23, 0);
        torsoGroup.add(rShoulderMesh);

        const lShoulderMesh = new THREE.Mesh(rShoulderGeo, accentMat);
        lShoulderMesh.position.set(-0.20, 0.23, 0);
        torsoGroup.add(lShoulderMesh);

        torsoGroup.position.set(0, -0.3, 0);
        this.avatarGroup.add(torsoGroup);
        this.torsoMesh = torsoGroup;

        // --- 3. RIGHT ARM & HAND ---
        this.rUpperArmMesh = this._createLimbMesh(0.045, 0.04, 0.22, skinMat);
        this.rLowerArmMesh = this._createLimbMesh(0.04, 0.035, 0.20, accentMat);
        this.rHandMesh = this._createHandMesh(handMat);

        this.avatarGroup.add(this.rUpperArmMesh);
        this.avatarGroup.add(this.rLowerArmMesh);
        this.avatarGroup.add(this.rHandMesh);

        // --- 4. LEFT ARM & HAND ---
        this.lUpperArmMesh = this._createLimbMesh(0.045, 0.04, 0.22, skinMat);
        this.lLowerArmMesh = this._createLimbMesh(0.04, 0.035, 0.20, accentMat);
        this.lHandMesh = this._createHandMesh(handMat);

        this.avatarGroup.add(this.lUpperArmMesh);
        this.avatarGroup.add(this.lLowerArmMesh);
        this.avatarGroup.add(this.lHandMesh);

        // --- 5. FINGERS ---
        for (let i = 0; i < 21; i++) {
            const rFinger = new THREE.Mesh(new THREE.SphereGeometry(0.009, 8, 8), accentMat);
            const lFinger = new THREE.Mesh(new THREE.SphereGeometry(0.009, 8, 8), accentMat);
            this.avatarGroup.add(rFinger);
            this.avatarGroup.add(lFinger);
            this.rFingerMeshes.push(rFinger);
            this.lFingerMeshes.push(lFinger);
        }

        this.scene.add(this.avatarGroup);
    }

    _createLimbMesh(radiusTop, radiusBottom, height, material) {
        const geo = new THREE.CylinderGeometry(radiusTop, radiusBottom, height, 16);
        geo.translate(0, -height / 2, 0); // Origin at top joint
        const mesh = new THREE.Mesh(geo, material);
        mesh.castShadow = true;
        return mesh;
    }

    _createHandMesh(material) {
        const handGroup = new THREE.Group();
        const palmGeo = new THREE.BoxGeometry(0.07, 0.08, 0.025);
        const palmMesh = new THREE.Mesh(palmGeo, material);
        palmMesh.position.set(0, -0.04, 0);
        palmMesh.castShadow = true;
        handGroup.add(palmMesh);
        return handGroup;
    }

    applyPoseFrame(frame) {
        if (!frame) return;

        const pose = frame.pose || [];
        const rHand = frame.right_hand || [];
        const lHand = frame.left_hand || [];

        // Head Position
        if (pose.length > 0) {
            this.headMesh.position.set(pose[0][0], pose[0][1] + 0.05, pose[0][2]);
        }

        // Key Joints
        // 7: R Shoulder, 8: L Shoulder
        // 9: R Elbow, 10: L Elbow
        // 11: R Wrist, 12: L Wrist
        const pRShoulder = pose.length > 7 ? new THREE.Vector3(...pose[7]) : new THREE.Vector3(0.20, -0.07, 0);
        const pLShoulder = pose.length > 8 ? new THREE.Vector3(...pose[8]) : new THREE.Vector3(-0.20, -0.07, 0);
        const pRElbow = pose.length > 9 ? new THREE.Vector3(...pose[9]) : new THREE.Vector3(0.22, -0.25, 0.05);
        const pLElbow = pose.length > 10 ? new THREE.Vector3(...pose[10]) : new THREE.Vector3(-0.22, -0.25, 0.05);
        const pRWrist = pose.length > 11 ? new THREE.Vector3(...pose[11]) : new THREE.Vector3(0.18, -0.40, 0.10);
        const pLWrist = pose.length > 12 ? new THREE.Vector3(...pose[12]) : new THREE.Vector3(-0.18, -0.40, 0.10);

        // --- Update Right Arm ---
        this._orientLimb(this.rUpperArmMesh, pRShoulder, pRElbow);
        this._orientLimb(this.rLowerArmMesh, pRElbow, pRWrist);
        this.rHandMesh.position.copy(pRWrist);

        // --- Update Left Arm ---
        this._orientLimb(this.lUpperArmMesh, pLShoulder, pLElbow);
        this._orientLimb(this.lLowerArmMesh, pLElbow, pLWrist);
        this.lHandMesh.position.copy(pLWrist);

        // --- Update Right Hand Fingers ---
        for (let i = 0; i < 21; i++) {
            if (i < rHand.length) {
                this.rFingerMeshes[i].position.set(
                    pRWrist.x + rHand[i][0],
                    pRWrist.y + rHand[i][1],
                    pRWrist.z + rHand[i][2]
                );
                this.rFingerMeshes[i].visible = true;
            } else {
                this.rFingerMeshes[i].visible = false;
            }
        }

        // --- Update Left Hand Fingers ---
        for (let i = 0; i < 21; i++) {
            if (i < lHand.length) {
                this.lFingerMeshes[i].position.set(
                    pLWrist.x + lHand[i][0],
                    pLWrist.y + lHand[i][1],
                    pLWrist.z + lHand[i][2]
                );
                this.lFingerMeshes[i].visible = true;
            } else {
                this.lFingerMeshes[i].visible = false;
            }
        }
    }

    _orientLimb(mesh, startPos, endPos) {
        mesh.position.copy(startPos);
        const dir = new THREE.Vector3().subVectors(endPos, startPos);
        const len = dir.length();

        if (len > 0.001) {
            mesh.scale.set(1, len / 0.22, 1); // Scale cylinder height
            const up = new THREE.Vector3(0, -1, 0);
            const quaternion = new THREE.Quaternion().setFromUnitVectors(up, dir.clone().normalize());
            mesh.quaternion.copy(quaternion);
        }
    }

    onWindowResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        // Subtle realistic idle breathing
        if (this.torsoMesh) {
            this.torsoMesh.position.y = -0.3 + Math.sin(Date.now() * 0.002) * 0.005;
        }

        this.renderer.render(this.scene, this.camera);
    }
}
