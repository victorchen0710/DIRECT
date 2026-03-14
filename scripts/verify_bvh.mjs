import fs from "fs";
import * as THREE from "three";
import { BVHLoader } from "three/examples/jsm/loaders/BVHLoader.js";

const file = process.argv[2];
if (!file) {
  console.error("Usage: node scripts/verify_bvh.mjs <path/to.bvh>");
  process.exit(1);
}

const text = fs.readFileSync(file, "utf-8");
const loader = new BVHLoader();
const res = loader.parse(text);
const { skeleton, clip } = res;
const root = skeleton.bones[0];

const mixer = new THREE.AnimationMixer(root);
const action = mixer.clipAction(clip);
action.play();

const times = clip.tracks[0].times;
const frameTime = times.length > 1 ? times[1] - times[0] : 0.066667;

const interesting = [
  "Hips",
  "LeftUpLeg",
  "RightUpLeg",
  "LeftLeg",
  "RightLeg",
  "LeftFoot",
  "RightFoot",
  "LeftToeBase",
  "RightToeBase",
  "Neck",
  "Head",
];

function sample(frameIdx) {
  const t = frameIdx * frameTime;
  mixer.setTime(t);
  root.updateMatrixWorld(true);
  skeleton.bones.forEach((b) => b.updateMatrixWorld(true));

  const out = {};
  for (const name of interesting) {
    const bone = skeleton.bones.find((b) => b.name === name);
    if (!bone) continue;
    const v = new THREE.Vector3();
    bone.getWorldPosition(v);
    out[name] = v.toArray().map((x) => +x.toFixed(4));
  }
  const minZ = Math.min(...Object.values(out).map((v) => v[2]));
  console.log(`frame ${frameIdx} (t=${t.toFixed(3)}s) minZ=${minZ.toFixed(3)}`);
  console.log(JSON.stringify(out, null, 2));
}

sample(0);
sample(Math.min(10, Math.max(1, times.length - 1)));
