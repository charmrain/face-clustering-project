import os, shutil

src_root = r"D:\project\photo\json\m6_json"                        # where chine-Scene-001 etc. live
dst_root = r"D:\project\photo\json\m6_collected_images"       # one big folder for all photos
os.makedirs(dst_root, exist_ok=True)

for scene in os.listdir(src_root):
    scene_path = os.path.join(src_root, scene, "images")
    if not os.path.isdir(scene_path):
        continue
    for fname in os.listdir(scene_path):
        src = os.path.join(scene_path, fname)
        if not os.path.isfile(src):
            continue
        new_name = f"{scene}_{fname}"    # add source folder name
        dst = os.path.join(dst_root, new_name)
        shutil.copy(src, dst)

print("All images gathered into:", dst_root)
