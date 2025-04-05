from multiprocessing import Pool
import numpy as np
import subprocess

PHYSIO_NET_ROOT_DIR = "/physionet"


def runcmd(cmd, verbose=False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_file(cxr_study_map, cxr_record_map, paths, i):

    k = 0
    for p in paths:
        if k < 3:
            val = cxr_study_map[p]
            uri = cxr_record_map[(val["subject_id"], val["study_id"])]
            runcmd("wget -r -N -c -np --user <USER> --password <PASSWORD> https://physionet.org/files/mimic-cxr/2.1.0/"
                   + uri)
            print("downloaded:", uri, " - process:", i)
        k += 1


if __name__ == "__main__":
    cpu_count = 4

    source_path = f"{PHYSIO_NET_ROOT_DIR}/ap_views.txt"
    source_file = open(source_path, encoding="utf-8")
    contents = source_file.read().splitlines()
    source_file.close()
    cxr_study_list = dict()
    with open(f"{PHYSIO_NET_ROOT_DIR}/files/mimic-cxr/2.1.0/cxr-study-list.csv", encoding="utf-8") as cxr_study_file:
        lines = cxr_study_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_study_list[items[2]] = {"subject_id": items[0], "study_id": items[1]}

    cxr_record_list = dict()
    with open(f"{PHYSIO_NET_ROOT_DIR}/files/mimic-cxr/2.1.0/cxr-record-list.csv", encoding="utf-8") as cxr_record_file:
        lines = cxr_record_file.readlines()
        for line in lines:
            items = line.strip().split(",")
            cxr_record_list[(items[0], items[1])] = items[3]

    pool = Pool(processes=cpu_count)
    content_chunks = np.array_split(contents, cpu_count)
    with Pool(cpu_count) as pool:
        results = pool.starmap(get_file, [(cxr_study_list, cxr_record_list, chunk, i) for i, chunk in enumerate(content_chunks)])

    pool.close()