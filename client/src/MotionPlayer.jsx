import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { BVHLoader } from "three/examples/jsm/loaders/BVHLoader.js";

export default function MotionPlayer({ bvhUrl, audioRef }) {
  const mountRef = useRef(null);

  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);

  const mixerRef = useRef(null);
  const clipRef = useRef(null);
  const clockRef = useRef(new THREE.Clock());

  const helperRef = useRef(null);
  const bvhGroupRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    console.log("[MotionPlayer] mounted. bvhUrl =", bvhUrl);

    // 容器尺寸必须非 0，否则渲染会异常
    const w0 = mount.clientWidth;
    const h0 = mount.clientHeight;
    console.log("[MotionPlayer] mount size:", w0, h0);

    // ---- Scene ----
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0b0b);
    sceneRef.current = scene;

    // ---- Camera ----
    const camera = new THREE.PerspectiveCamera(45, (w0 || 1) / (h0 || 1), 0.01, 1e7);
    camera.position.set(0, 120, 260);
    camera.lookAt(0, 0, 80);
    cameraRef.current = camera;
    // camera.up.set(0, 0, 1);

    // ---- Renderer ----
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w0 || 1, h0 || 1);
    rendererRef.current = renderer;

    // 确保 canvas 填满父容器
    renderer.domElement.style.position = "absolute";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";

    mount.appendChild(renderer.domElement);

    // ---- Controls (交互) ----
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 80);
    controls.update();
    controlsRef.current = controls;

    // ---- Lights ----
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(200, 300, 150);
    scene.add(dir);

    // ---- Grid ----
    const grid = new THREE.GridHelper(800, 40);
    // grid.rotation.x = Math.PI / 2;
    // grid.position.z = 0;
    scene.add(grid);

    // ---- BVH Group ----
    const bvhGroup = new THREE.Group();
    scene.add(bvhGroup);
    bvhGroupRef.current = bvhGroup;

    let cancelled = false;

    async function loadBVH() {
      // 清理旧对象
      if (helperRef.current) {
        scene.remove(helperRef.current);
        helperRef.current = null;
      }
      bvhGroup.clear();
      mixerRef.current = null;
      clipRef.current = null;

      console.log("[MotionPlayer] fetching BVH:", bvhUrl);
      const text = await fetch(bvhUrl).then((r) => r.text());

      console.log("[MotionPlayer] BVH head:\n", text.slice(0, 120));
      console.log("[MotionPlayer] BVH length:", text.length);

      if (!text.includes("HIERARCHY") || !text.includes("MOTION")) {
        throw new Error("Not BVH text: missing HIERARCHY/MOTION");
      }
      if (cancelled) return;

      const loader = new BVHLoader();
      const result = loader.parse(text);

      console.log("[MotionPlayer] parsed:", {
        bones: result.skeleton?.bones?.length,
        duration: result.clip?.duration,
      });

      const skeleton = result.skeleton;
      const clip = result.clip;
      clipRef.current = clip;

      const rootBone = skeleton.bones[0];
      bvhGroup.add(rootBone);

      // === 复现 Blender: axis_forward='-Z', axis_up='Y' ===
      // M = Rx(-90°) * Rz(180°)
      // const m = new THREE.Matrix4()
      //   .makeRotationX(-Math.PI / 2)
      //   .multiply(new THREE.Matrix4().makeRotationZ(Math.PI));

      // bvhGroup.applyMatrix4(m);
      // bvhGroup.updateMatrixWorld(true);
      // !!! 删掉你原来的 bvhGroup.rotation.x = -Math.PI / 2; 这句（不要重复旋转）

      // 画骨架
      const helper = new THREE.SkeletonHelper(rootBone);
      helper.skeleton = skeleton;
      helper.material.depthTest = false;
      helper.renderOrder = 999;
      scene.add(helper);
      helperRef.current = helper;

      // Mixer
      const mixer = new THREE.AnimationMixer(rootBone);
      mixerRef.current = mixer;
      mixer.clipAction(clip).play();

      // === Grounding（Z-up）：把最低点抬到 z=0 ===
      helper.updateMatrixWorld(true);
      {
        const groundedBox = new THREE.Box3().setFromObject(helper);
        const minZ = groundedBox.min.z;
        bvhGroup.position.z -= minZ;
        bvhGroup.updateMatrixWorld(true);
        helper.updateMatrixWorld(true);
      }

      // === 重新计算 box/center（必须在 grounding 后）===
      const box = new THREE.Box3().setFromObject(helper);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;

      // === Fit camera（Z-up：高度看 center.z）===
      controls.target.copy(center);
      controls.update();

      const fov = (camera.fov * Math.PI) / 180;
      let dist = (maxDim / 2) / Math.tan(fov / 2);
      dist *= 2.0;

      camera.near = 0.01;
      camera.far = 1e7;
      camera.updateProjectionMatrix();

      // 让相机从 “-Y 方向” 看向中心，Z 方向抬高一点（更像 Blender 视角）
      camera.position.set(center.x, center.y - dist, center.z + maxDim * 0.6);
      camera.lookAt(center);
      camera.updateProjectionMatrix();

      // 如果你发现“躺倒/旋转 90 度”，确认能看到后再试下面之一：
      // bvhGroup.rotation.x = -Math.PI / 2;
      // bvhGroup.rotation.y = Math.PI;
    }

    loadBVH().catch((e) => {
      console.error("[MotionPlayer] loadBVH failed:", e);
    });

    // ---- Render loop ----
    let rafId = 0;
    function animate() {
      rafId = requestAnimationFrame(animate);

      controls.update();

      const mixer = mixerRef.current;
      const clip = clipRef.current;

      if (mixer && clip) {
        const tAudio = audioRef?.current?.currentTime;
        if (typeof tAudio === "number" && !Number.isNaN(tAudio)) {
          const t = Math.max(0, Math.min(tAudio, clip.duration));
          mixer.setTime(t);
        } else {
          mixer.update(clockRef.current.getDelta());
        }
      }
      helperRef.current?.updateMatrixWorld(true);
      renderer.render(scene, camera);
    }
    animate();

    // ---- Resize ----
    const onResize = () => {
      if (!mountRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = mountRef.current.clientWidth || 1;
      const h = mountRef.current.clientHeight || 1;
      rendererRef.current.setSize(w, h);
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(rafId);

      try {
        controls.dispose();
        renderer.dispose();
        if (renderer.domElement && mount.contains(renderer.domElement)) {
          mount.removeChild(renderer.domElement);
        }
      } catch {}
    };
  }, [bvhUrl, audioRef]);

  // 关键：必须 absolute 填满 .preview-window（它是 position: relative）
  return <div ref={mountRef} className="motion-player-mount" />;
}
