def ensure_coco_dataset_compat(coco):
    dataset = getattr(coco, "dataset", None)
    if not isinstance(dataset, dict):
        return coco

    # Some hand-written COCO json files omit optional top-level keys that
    # pycocotools.loadRes still assumes exist.
    dataset.setdefault("info", {})
    dataset.setdefault("licenses", [])
    return coco
