import argparse
import numpy as np

parser = argparse.ArgumentParser(prog="eval segments")
parser.add_argument("-t", "--target", help="target segment file to be evaluated")
parser.add_argument("-r", "--reference", help="reference segment file")


def evaluate(target_fn, reference_fn):
    tgt = np.loadtxt(target_fn, dtype=np.int32)
    ref = np.loadtxt(reference_fn, dtype=np.int32)
    tgt_parts = np.unique(tgt)
    ref_parts = np.unique(ref)

    assert tgt_parts.shape == ref_parts.shape

    iou_total = 0
    part_num = ref_parts.shape[0] - 1

    if part_num == 0:
        return 0

    for pid in np.unique(ref).tolist():
        if pid == 0:  # background
            continue

        tgt_area = tgt == pid
        ref_area = ref == pid

        union_area = np.logical_or(tgt_area, ref_area)
        intersect_area = np.logical_and(tgt_area, ref_area)

        iou_total += intersect_area.sum() / (union_area.sum())

    return iou_total / part_num


if __name__ == "__main__":
    args = parser.parse_args()
    print(evaluate(target_fn=args.target, reference_fn=args.reference))
