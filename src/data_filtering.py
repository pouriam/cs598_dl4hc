import re
import os

ap_pattern = re.compile(
    r"(AP\s*chest|Portable.*AP|anteroposterior|single frontal view|portable (semi-upright|semi-erect))|AP VIEW|AP ONLY",
    re.IGNORECASE)

PHYSIO_NET_ROOT_DIR = "/physionet"

if __name__ == "__main__":
    dataset_root_dir = f"{PHYSIO_NET_ROOT_DIR}/files/mimic-cxr/2.1.0/mimic-cxr-reports/"
    output_file = open(f"{PHYSIO_NET_ROOT_DIR}/ap_views.txt", "w", encoding="utf-8")
    c = 0
    for (dirpath, dirnames, filenames) in os.walk(dataset_root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                with open(os.path.join(dirpath, filename), encoding="utf-8") as f:
                    content = f.read()
                    c += 1
                    if ap_pattern.search(content):
                        output_file.write(os.path.join(dirpath, filename).replace(dataset_root_dir, "")
                                          .replace("\\", "/"))
                        output_file.write("\n")

    print(f"traversed {c} files")
    output_file.close()
