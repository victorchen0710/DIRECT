import sys
import bpy


def main(argv):
    if "--" not in argv or len(argv) - argv.index("--") < 3:
        print("Usage: blender -b -P backend/blender_script.py -- in.bvh out.glb")
        return

    args = argv[argv.index("--") + 1 :]
    src, dst = args[0], args[1]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_anim.bvh(
        filepath=src,
        global_scale=1.0,
        axis_forward="-Z",
        axis_up="Y",
    )

    bpy.context.scene.frame_set(1)
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_yup=True)
    print(f"[blender_script] Exported {dst}")


if __name__ == "__main__":
    main(sys.argv)
