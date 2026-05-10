import csv
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

annotations_path = Path("data/coco20k/annotations/captions_train2014.json")
output_dir = Path("data/coco20k")
images_dir = output_dir / "images"
captions_path = output_dir / "captions.txt"

num_images = 20000
seed = 42

images_dir.mkdir(parents=True, exist_ok=True)

with open(annotations_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images_by_id = {img["id"]: img for img in data["images"]}

captions_by_image = {}
for ann in data["annotations"]:
    captions_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

valid_ids = sorted(set(images_by_id) & set(captions_by_image))

random.seed(seed)
selected_ids = sorted(random.sample(valid_ids, num_images))

with open(captions_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])

    for image_id in selected_ids:
        image_name = images_by_id[image_id]["file_name"]
        for caption in captions_by_image[image_id]:
            writer.writerow([image_name, caption])

downloaded = 0


def download_file(url, output_path):
    try:
        with urlopen(url, timeout=60) as response:
            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except (HTTPError, URLError, TimeoutError, PermissionError, OSError) as e:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        raise e

for i, image_id in enumerate(selected_ids, start=1):
    image = images_by_id[image_id]
    image_name = image["file_name"]
    url = image.get("coco_url") or f"http://images.cocodataset.org/train2014/{image_name}"
    output_path = images_dir / image_name

    if output_path.exists():
        downloaded += 1
    else:
        for attempt in range(3):
            try:
                download_file(url, output_path)
                downloaded += 1
                break
            except (HTTPError, URLError, TimeoutError, PermissionError, OSError) as e:
                if attempt == 2:
                    print(f"Failed: {image_name} | {e}")
                else:
                    time.sleep(2)

    if i % 100 == 0:
        print(f"Downloaded {downloaded}/{i}")

print("Done")
print("Images:", images_dir)
print("Captions:", captions_path)
print("Selected images:", len(selected_ids))
