import argparse
import json
import os
import subprocess
import sys

DEFAULT_CONFIG_PATH = "configs/train_experiment_config.json"
DEFAULT_PRESETS_PATH = "configs/experiment_presets.json"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_dataset_path(path_value, root_prefix):
    path_value = str(path_value or "").strip()
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value.replace("\\", "/")
    if root_prefix:
        root_parts = [p for p in root_prefix.replace("\\", "/").split("/") if p]
        path_parts = [p for p in path_value.replace("\\", "/").split("/") if p]
        max_overlap = min(len(root_parts), len(path_parts))
        for overlap in range(max_overlap, 0, -1):
            if root_parts[-overlap:] == path_parts[:overlap]:
                tail = path_parts[overlap:]
                joined = os.path.join(root_prefix, *tail) if tail else root_prefix
                return joined.replace("\\", "/")
        return os.path.join(root_prefix, path_value).replace("\\", "/")
    return path_value.replace("\\", "/")


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in ("1", "true", "yes", "y", "on")


def value_or_default(value, default):
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return value


def normalize_box_precision(value):
    text = str(value_or_default(value, "high")).strip().lower()
    if text in ("truncate", "truncated", "int", "legacy"):
        return "truncate"
    return "high"


def run_command(cmd, env=None, dry_run=False):
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env, cwd=PROJECT_ROOT)



def resolve_pretrained(network_name, configured_model_path, network_alias, network_pretrained_map):
    net_key = network_alias.get(str(network_name).lower(), str(network_name).lower())
    default_path = network_pretrained_map.get(net_key, "")

    if configured_model_path is not None:
        text = str(configured_model_path).strip()
        if text == "":
            return default_path, default_path, net_key
        if text.lower() in ("none", "null", "no", "random", "scratch"):
            return "", default_path, net_key
        return text, default_path, net_key

    return default_path, default_path, net_key

def resolve_train_hparams(network_name, config, network_alias, network_train_defaults):
    net_key = network_alias.get(str(network_name).lower(), str(network_name).lower())
    defaults = network_train_defaults.get(net_key)
    if defaults is None and str(net_key).lower().startswith("acm"):
        fallback_key = "acm_unet" if "unet" in str(net_key).lower() else "acm_fpn"
        defaults = network_train_defaults.get(
            fallback_key,
            {
                "optimizer_type": "adagrad",
                "init_lr": 0.01,
                "min_lr_ratio": 0.0001,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        )
    if defaults is None:
        defaults = network_train_defaults.get("sstnet", {})

    optimizer_type = str(config.get("optimizer_type") or defaults.get("optimizer_type", "sgd")).lower()
    init_lr = float(config.get("init_lr") or defaults.get("init_lr", 1e-2))
    min_lr = float(config.get("min_lr") or (init_lr * float(defaults.get("min_lr_ratio", 0.01))))
    momentum = float(config.get("momentum") or defaults.get("momentum", 0.937))
    weight_decay = float(config.get("weight_decay") or defaults.get("weight_decay", 5e-4))
    return net_key, optimizer_type, init_lr, min_lr, momentum, weight_decay


def is_acm_network(net_key):
    return str(net_key).lower() in ("acm_fpn", "acm_unet", "acm_unet_saliency")


def is_dna_network(net_key):
    return str(net_key).lower() in ("dnanet", "dnanet_saliency")


def is_dna_saliency_network(net_key):
    return str(net_key).lower() == "dnanet_saliency"


def is_alc_network(net_key):
    return str(net_key).lower() in ("alcnet", "alcnet_saliency")


def is_alc_saliency_network(net_key):
    return str(net_key).lower() == "alcnet_saliency"


def is_uiu_network(net_key):
    return str(net_key).lower() in ("uiunet", "uiunet_saliency")


def is_uiu_saliency_network(net_key):
    return str(net_key).lower() == "uiunet_saliency"


def is_sctrans_network(net_key):
    return str(net_key).lower() in ("sctransnet", "sctransnet_det")


def is_dqaligner_network(net_key):
    return str(net_key).lower() in ("dqaligner", "dqaligner_det", "dqaligner_saliency", "dqaligner_saliency_det")


def resolve_train_script(config, preset, net_key, network_train_scripts):
    override = config.get("train_script_override")
    if override:
        return override
    network_specific = network_train_scripts.get(net_key, "")
    if network_specific:
        return network_specific
    if is_acm_network(net_key):
        return "train/train_ACM_det.py"
    if is_dna_network(net_key):
        return "train/train_DNANet_det.py"
    if is_uiu_network(net_key):
        return "train/train_UIUNet_det.py"
    if is_sctrans_network(net_key):
        return "train/train_SCTransNet_det.py"
    if is_alc_network(net_key):
        return "train/train_ALCNet_det.py"
    return preset["train_script"]


def resolve_dataset_override(config, key, dataset_name, default_value):
    overrides = config.get(f"{key}_by_dataset", {})
    if isinstance(overrides, dict):
        for name in (dataset_name, str(dataset_name).upper(), str(dataset_name).lower()):
            if name in overrides and str(overrides[name]).strip() != "":
                return overrides[name]
    return config.get(key, default_value)


def run_train(config, presets, dry_run=False):
    dataset_presets = presets["dataset_presets"]
    network_alias = presets.get("network_alias", {})
    network_pretrained_map = presets.get("network_pretrained_map", {})
    network_train_defaults = presets.get("network_train_defaults", {})
    network_presets = presets.get("network_presets", {})
    network_train_scripts = presets.get("network_train_scripts", {})
    global_cfg = presets.get("global", {})

    dataset_name = config["dataset"]
    preset = dataset_presets.get(dataset_name)
    if preset is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(dataset_presets.keys())}")

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT if not env.get("PYTHONPATH") else f"{PROJECT_ROOT}{os.pathsep}{env['PYTHONPATH']}"

    network_name = config.get("network", "sstnet")
    train_model_path, bound_default, net_key = resolve_pretrained(
        network_name, config.get("train_model_path", ""), network_alias, network_pretrained_map
    )
    if net_key not in network_pretrained_map:
        raise ValueError(
            "Unknown network '%s'. Use one of: %s"
            % (network_name, ", ".join(sorted(network_pretrained_map.keys())))
        )

    _, optimizer_type, init_lr, min_lr, momentum, weight_decay = resolve_train_hparams(
        network_name, config, network_alias, network_train_defaults
    )

    train_txt_path = config.get("train_txt_path") or preset.get("default_train_txt", "")
    val_txt_path = config.get("val_txt_path") or preset.get("default_val_txt", "")
    save_root = config.get("train_output_root", "logs")
    dataset_root_prefix = (
        str(config.get("dataset_root_prefix", "")).strip()
        or str(global_cfg.get("dataset_root_prefix", "")).strip()
    )

    train_script = resolve_train_script(config, preset, net_key, network_train_scripts)
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"Train script not found: {train_script}")

    if is_acm_network(net_key) and train_model_path:
        model_path_exists = os.path.exists(train_model_path)
        if (not model_path_exists) and (not os.path.isabs(train_model_path)):
            model_path_exists = os.path.exists(os.path.join(PROJECT_ROOT, train_model_path))
        if not model_path_exists:
            print(f"ACM pretrained not found, train from scratch: {train_model_path}")
            train_model_path = ""

    print(f"Network: {net_key} | Pretrained: {train_model_path or '(none)'}")
    if bound_default and train_model_path != bound_default:
        print(f"Note: overriding bound default ({bound_default}) with {train_model_path}")
    print(
        "Train hparams:",
        f"optimizer={optimizer_type}, init_lr={init_lr}, min_lr={min_lr}, momentum={momentum}, weight_decay={weight_decay}",
    )
    print("Dataset root prefix:", dataset_root_prefix or "(none)")
    print("Train outputs root:", os.path.join(save_root, dataset_name, network_name))
    print("Train script:", train_script)

    env["NETWORK_NAME"] = network_name
    env["DATASET_NAME"] = dataset_name
    env["SAVE_ROOT"] = save_root
    if dataset_root_prefix:
        env["DATASET_ROOT_PREFIX"] = dataset_root_prefix.replace("\\", "/")
    if train_model_path:
        env["MODEL_PATH"] = train_model_path
    env["OPTIMIZER_TYPE"] = optimizer_type
    env["INIT_LR"] = str(init_lr)
    env["MIN_LR"] = str(min_lr)
    env["MOMENTUM"] = str(momentum)
    env["WEIGHT_DECAY"] = str(weight_decay)
    env["EVAL_PERIOD"] = str(int(config.get("eval_period", 10)))
    eval_confidence_global = float(config.get("eval_confidence", 0.001))
    eval_nms_iou_global = float(config.get("eval_nms_iou", 0.6))
    env["EVAL_CONFIDENCE"] = str(eval_confidence_global)
    env["EVAL_NMS_IOU"] = str(eval_nms_iou_global)
    env["EVAL_MAX_BOXES"] = str(int(config.get("eval_max_boxes", 100)))
    default_predict = preset.get("default_predict", {})
    eval_json_path_global = str(config.get("eval_json_path") or default_predict.get("json_path", ""))
    eval_image_root_global = resolve_dataset_path(
        str(config.get("eval_image_root") or default_predict.get("dataset_img_path", "")),
        dataset_root_prefix,
    )
    eval_flag_global = bool(config.get("eval_flag", True))
    env["EVAL_FLAG"] = "1" if eval_flag_global else "0"
    env["EVAL_JSON_PATH"] = eval_json_path_global
    env["EVAL_IMAGE_ROOT"] = eval_image_root_global
    if "eval_letterbox" in config:
        env["EVAL_LETTERBOX"] = "1" if bool(config.get("eval_letterbox")) else "0"
    if config.get("letterbox_train") is not None:
        letterbox_train = as_bool(config.get("letterbox_train"), False)
        letterbox_value = "1" if letterbox_train else "0"
        env["DATA_DETLAB_LETTERBOX_TRAIN"] = letterbox_value
        env["ACM_LETTERBOX_TRAIN"] = letterbox_value
        env["ALC_LETTERBOX_TRAIN"] = letterbox_value
        env["UIU_LETTERBOX_TRAIN"] = letterbox_value
        print(
            "Train letterbox:",
            "enabled (force random=False, mosaic=False, mixup=False in utils/acm/data_detlab.py)"
            if letterbox_train
            else "disabled",
        )
    if config.get("box_coord_type") is not None:
        box_coord_type = str(config.get("box_coord_type") or "float").strip().lower()
        if box_coord_type in ("int", "integer"):
            box_coord_type = "int"
        else:
            box_coord_type = "float"
        env["DETLAB_BOX_COORD_TYPE"] = box_coord_type
        env["DATA_DETLAB_BOX_COORD_TYPE"] = box_coord_type
        env["ACM_BOX_COORD_TYPE"] = box_coord_type
        env["ALC_BOX_COORD_TYPE"] = box_coord_type
        env["UIU_BOX_COORD_TYPE"] = box_coord_type
        print("DETLAB box coord type:", box_coord_type)
    box_precision = normalize_box_precision(config.get("box_precision", "high"))
    if config.get("box_geometry_dtype") is not None:
        box_precision = normalize_box_precision(config.get("box_geometry_dtype"))
    if config.get("sequence_box_mode") is not None:
        box_precision = normalize_box_precision(config.get("sequence_box_mode"))
    box_geometry_dtype = "int" if box_precision == "truncate" else "float"
    sequence_box_mode = "legacy" if box_precision == "truncate" else "float-copy"
    env["DETLAB_BOX_GEOMETRY_DTYPE"] = box_geometry_dtype
    env["DATA_DETLAB_BOX_GEOMETRY_DTYPE"] = box_geometry_dtype
    env["ACM_BOX_GEOMETRY_DTYPE"] = box_geometry_dtype
    env["ALC_BOX_GEOMETRY_DTYPE"] = box_geometry_dtype
    env["UIU_BOX_GEOMETRY_DTYPE"] = box_geometry_dtype
    env["SEQUENCE_BOX_MODE"] = sequence_box_mode
    env["SEQUENCE_DATASET_BOX_MODE"] = sequence_box_mode
    print(
        "Box precision:",
        "%s (single-frame geometry=%s, sequence=%s)" % (box_precision, box_geometry_dtype, sequence_box_mode),
    )
    if train_txt_path:
        env["TRAIN_ANNOTATION_PATH"] = train_txt_path
    if val_txt_path:
        env["VAL_ANNOTATION_PATH"] = val_txt_path

    if is_dqaligner_network(net_key):
        dq_preset = network_presets.get(net_key, network_presets.get("dqaligner", {}))
        dq_batch = int(config.get("dqaligner_batch_size", dq_preset.get("batch_size", 1)))
        dq_epochs = int(config.get("dqaligner_epochs", dq_preset.get("epochs", config.get("epochs", 100))))
        dq_freeze_epochs = int(config.get("dqaligner_freeze_epochs", dq_preset.get("freeze_epochs", dq_epochs)))
        dq_save_period = int(config.get("dqaligner_save_period", dq_preset.get("save_period", 10)))
        env["SEQUENCE_BATCH_SIZE"] = str(dq_batch)
        env["SEQUENCE_FREEZE_BATCH_SIZE"] = str(dq_batch)
        env["SEQUENCE_UNFREEZE_BATCH_SIZE"] = str(dq_batch)
        env["SEQUENCE_EPOCHS"] = str(dq_epochs)
        env["SEQUENCE_FREEZE_EPOCHS"] = str(dq_freeze_epochs)
        env["SEQUENCE_SAVE_PERIOD"] = str(dq_save_period)
        print(
            "DQAligner train:",
            f"mode={dq_preset.get('det_mode', 'feature')}, num_frame={dq_preset.get('num_frame', 5)}, batch_size={dq_batch}, epochs={dq_epochs}, dataset=sequence-detlab",
        )

    if not (is_uiu_network(net_key) or is_dna_network(net_key) or is_acm_network(net_key) or is_alc_network(net_key) or is_sctrans_network(net_key)):
        print(
            "Sequence eval:",
            f"enabled={eval_flag_global}, period={int(config.get('eval_period', 10))}, confidence={eval_confidence_global}, nms_iou={eval_nms_iou_global}, json={eval_json_path_global or '(none)'}, image_root={eval_image_root_global or '(none)'}",
        )

    if is_uiu_network(net_key):
        uiu_preset = network_presets.get(net_key, network_presets.get("uiunet", {}))
        default_predict = preset.get("default_predict", {})
        uiu_det_mode = str(config.get("uiu_det_mode", uiu_preset.get("det_mode", "")) or "")
        if not uiu_det_mode:
            uiu_det_mode = "saliency" if is_uiu_saliency_network(net_key) else "feature"

        env["UIU_FUSE_MODE"] = str(config.get("uiu_fuse_mode", uiu_preset.get("fuse_mode", "AsymBi")))
        env["UIU_DET_MODE"] = uiu_det_mode
        env["UIU_IMAGE_ROOT"] = resolve_dataset_path(
            str(config.get("uiu_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        env["UIU_BASE_SIZE"] = str(int(config.get("uiu_base_size", uiu_preset.get("base_size", 512))))
        env["UIU_BATCH_SIZE"] = str(int(config.get("uiu_batch_size", uiu_preset.get("batch_size", 8))))
        env["UIU_EPOCHS"] = str(int(config.get("uiu_epochs", uiu_preset.get("epochs", 100))))
        env["UIU_WARM_UP_EPOCHS"] = str(int(config.get("uiu_warm_up_epochs", uiu_preset.get("warm_up_epochs", 0))))
        env["UIU_NUM_CLASSES"] = str(int(config.get("uiu_num_classes", uiu_preset.get("num_classes", 1))))
        env["UIU_STRIDE"] = str(config.get("uiu_stride", uiu_preset.get("stride", "8")))
        env["UIU_NUM_WORKERS"] = str(int(config.get("uiu_num_workers", uiu_preset.get("num_workers", 2))))
        env["UIU_SAVE_PERIOD"] = str(int(config.get("uiu_save_period", uiu_preset.get("save_period", 10))))
        uiu_fp16 = bool(config.get("uiu_fp16", uiu_preset.get("fp16", False)))
        env["UIU_FP16"] = "1" if uiu_fp16 else "0"
        env["UIU_GRAD_CLIP_NORM"] = str(float(config.get("uiu_grad_clip_norm", uiu_preset.get("grad_clip_norm", 1.0))))
        env["UIU_SKIP_INVALID_GRAD"] = str(int(config.get("uiu_skip_invalid_grad", 1)))

        eval_flag = bool(config.get("uiu_eval_flag", config.get("eval_flag", True)))
        eval_period = int(config.get("uiu_eval_period", config.get("eval_period", 10)))
        eval_json_path = str(config.get("uiu_eval_json_path") or default_predict.get("json_path", ""))
        eval_image_root = resolve_dataset_path(
            str(config.get("uiu_eval_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        eval_confidence = eval_confidence_global
        eval_nms_iou = eval_nms_iou_global

        env["UIU_EVAL_FLAG"] = "1" if eval_flag else "0"
        env["UIU_EVAL_PERIOD"] = str(eval_period)
        env["UIU_EVAL_JSON_PATH"] = eval_json_path
        env["UIU_EVAL_IMAGE_ROOT"] = eval_image_root
        env["UIU_EVAL_CONFIDENCE"] = str(eval_confidence)
        env["UIU_EVAL_NMS_IOU"] = str(eval_nms_iou)

        print(
            "UIUNet eval:",
            f"enabled={eval_flag}, period={eval_period}, confidence={eval_confidence}, nms_iou={eval_nms_iou}, json={eval_json_path or '(none)'}, image_root={eval_image_root or '(none)'}",
        )
        print(
            "UIUNet train:",
            f"mode={uiu_det_mode}, base_size={env['UIU_BASE_SIZE']}, batch_size={env['UIU_BATCH_SIZE']}, stride={env['UIU_STRIDE']}, dataset=detlab, fp16={uiu_fp16}, grad_clip_norm={env['UIU_GRAD_CLIP_NORM']}, skip_invalid_grad={env['UIU_SKIP_INVALID_GRAD']}",
        )

    if is_dna_network(net_key):
        dna_preset = network_presets.get(net_key, network_presets.get("dnanet", {}))
        default_predict = preset.get("default_predict", {})
        dna_det_mode = str(config.get("dna_det_mode", dna_preset.get("det_mode", "")) or "")
        if not dna_det_mode:
            dna_det_mode = "saliency" if is_dna_saliency_network(net_key) else "feature"

        env["DNA_DET_MODE"] = dna_det_mode
        env["DNA_CHANNEL_SIZE"] = str(config.get("dna_channel_size", dna_preset.get("channel_size", "three")))
        env["DNA_BACKBONE"] = str(config.get("dna_backbone", dna_preset.get("backbone", "resnet_18")))
        env["DNA_NUM_CLASSES"] = str(int(config.get("dna_num_classes", dna_preset.get("num_classes", 1))))
        env["DNA_STRIDE"] = str(int(config.get("dna_stride", dna_preset.get("stride", 8))))
        dna_base_size = resolve_dataset_override(config, "dna_base_size", dataset_name, dna_preset.get("base_size", 512))
        dna_batch_size = resolve_dataset_override(config, "dna_batch_size", dataset_name, dna_preset.get("batch_size", 8))
        env["DNA_BASE_SIZE"] = str(int(dna_base_size))
        env["DNA_BATCH_SIZE"] = str(int(dna_batch_size))
        env["DNA_EPOCHS"] = str(int(config.get("dna_epochs", dna_preset.get("epochs", 100))))
        env["DNA_WARM_UP_EPOCHS"] = str(int(config.get("dna_warm_up_epochs", dna_preset.get("warm_up_epochs", 0))))
        env["DNA_NUM_WORKERS"] = str(int(config.get("dna_num_workers", dna_preset.get("num_workers", 2))))
        env["DNA_SAVE_PERIOD"] = str(int(config.get("dna_save_period", dna_preset.get("save_period", 10))))
        env["DNA_SUFFIX"] = str(config.get("dna_suffix", ".png"))
        env["DNA_MOSAIC"] = "1" if as_bool(config.get("dna_mosaic"), dna_preset.get("mosaic", True)) else "0"
        env["DNA_MIXUP"] = "1" if as_bool(config.get("dna_mixup"), dna_preset.get("mixup", True)) else "0"
        env["DNA_MOSAIC_PROB"] = str(float(config.get("dna_mosaic_prob", dna_preset.get("mosaic_prob", 0.5))))
        env["DNA_MIXUP_PROB"] = str(float(config.get("dna_mixup_prob", dna_preset.get("mixup_prob", 0.5))))
        env["DNA_SPECIAL_AUG_RATIO"] = str(float(config.get("dna_special_aug_ratio", dna_preset.get("special_aug_ratio", 0.7))))
        env["DNA_GRAD_CLIP_NORM"] = str(float(config.get("dna_grad_clip_norm", dna_preset.get("grad_clip_norm", 1.0))))
        env["DNA_SKIP_INVALID_GRAD"] = str(int(config.get("dna_skip_invalid_grad", dna_preset.get("skip_invalid_grad", 1))))
        env["DNA_EMA"] = "1" if as_bool(config.get("dna_ema"), dna_preset.get("ema", False)) else "0"
        env["DNA_FREEZE_TRAIN"] = "1" if as_bool(config.get("dna_freeze_train"), dna_preset.get("freeze_train", False)) else "0"
        env["DNA_FREEZE_EPOCHS"] = str(int(config.get("dna_freeze_epochs", dna_preset.get("freeze_epochs", 100))))
        env["DNA_LR_DECAY_TYPE"] = str(config.get("dna_lr_decay_type", dna_preset.get("lr_decay_type", "cos")))

        env["DNA_IMAGE_ROOT"] = resolve_dataset_path(
            str(config.get("dna_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        env["DNA_SOURCE_IMAGE_ROOT"] = str(config.get("dna_source_image_root", ""))

        eval_flag = bool(config.get("dna_eval_flag", config.get("eval_flag", True)))
        eval_period = int(config.get("dna_eval_period", config.get("eval_period", 10)))
        eval_json_path = str(config.get("dna_eval_json_path") or default_predict.get("json_path", ""))
        eval_image_root = resolve_dataset_path(
            str(config.get("dna_eval_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        eval_confidence = eval_confidence_global
        eval_nms_iou = eval_nms_iou_global

        env["DNA_EVAL_FLAG"] = "1" if eval_flag else "0"
        env["DNA_EVAL_PERIOD"] = str(eval_period)
        env["DNA_EVAL_JSON_PATH"] = eval_json_path
        env["DNA_EVAL_IMAGE_ROOT"] = eval_image_root
        env["DNA_EVAL_CONFIDENCE"] = str(eval_confidence)
        env["DNA_EVAL_NMS_IOU"] = str(eval_nms_iou)

        print(
            "DNANet eval:",
            f"enabled={eval_flag}, period={eval_period}, confidence={eval_confidence}, nms_iou={eval_nms_iou}, json={eval_json_path or '(none)'}, image_root={eval_image_root or '(none)'}",
        )
        print(
            "DNANet train:",
            f"mode={dna_det_mode}, base_size={env['DNA_BASE_SIZE']}, batch_size={env['DNA_BATCH_SIZE']}, stride={env['DNA_STRIDE']}, dataset=detlab, mosaic={env['DNA_MOSAIC']}, mixup={env['DNA_MIXUP']}, ema={env['DNA_EMA']}, lr_decay={env['DNA_LR_DECAY_TYPE']}, grad_clip_norm={env['DNA_GRAD_CLIP_NORM']}, skip_invalid_grad={env['DNA_SKIP_INVALID_GRAD']}",
        )

    if is_acm_network(net_key):
        acm_preset = network_presets.get(net_key, {})
        default_predict = preset.get("default_predict", {})
        acm_name = str(net_key).lower()
        default_acm_backbone = "UNet" if ("unet" in acm_name or "saliency" in acm_name) else "FPN"
        default_acm_det_mode = "saliency" if "saliency" in acm_name else "feature"

        acm_det_mode = str(config.get("acm_det_mode") or acm_preset.get("det_mode") or default_acm_det_mode)
        env["ACM_BACKBONE_MODE"] = str(config.get("acm_backbone_mode") or acm_preset.get("backbone_mode") or default_acm_backbone)
        env["ACM_DET_MODE"] = acm_det_mode
        env["ACM_FUSE_MODE"] = str(config.get("acm_fuse_mode") or acm_preset.get("fuse_mode", "AsymBi"))
        env["ACM_BLOCKS_PER_LAYER"] = str(int(config.get("acm_blocks_per_layer") or acm_preset.get("blocks_per_layer", 4)))
        env["ACM_IMAGE_ROOT"] = resolve_dataset_path(
            str(config.get("acm_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        env["ACM_BASE_SIZE"] = str(int(config.get("acm_base_size", 512)))
        env["ACM_BATCH_SIZE"] = str(int(config.get("acm_batch_size", 8)))
        env["ACM_EPOCHS"] = str(int(config.get("acm_epochs", 100)))
        env["ACM_WARM_UP_EPOCHS"] = str(int(config.get("acm_warm_up_epochs", 0)))
        env["ACM_NUM_CLASSES"] = str(int(config.get("acm_num_classes", 1)))
        env["ACM_STRIDE"] = str(int(config.get("acm_stride", acm_preset.get("stride", 8))))
        env["ACM_NUM_WORKERS"] = str(int(config.get("acm_num_workers", 2)))
        env["ACM_SAVE_EVERY"] = str(int(config.get("acm_save_every", acm_preset.get("save_period", 10))))
        env["ACM_RUN_NAME"] = str(config.get("acm_run_name", ""))
        acm_mosaic = as_bool(config.get("acm_mosaic"), acm_preset.get("mosaic", True))
        acm_mixup = as_bool(config.get("acm_mixup"), acm_preset.get("mixup", True))
        env["ACM_MOSAIC"] = "1" if acm_mosaic else "0"
        env["ACM_MIXUP"] = "1" if acm_mixup else "0"
        env["ACM_MOSAIC_PROB"] = str(float(config.get("acm_mosaic_prob", acm_preset.get("mosaic_prob", 0.5))))
        env["ACM_MIXUP_PROB"] = str(float(config.get("acm_mixup_prob", acm_preset.get("mixup_prob", 0.5))))
        env["ACM_SPECIAL_AUG_RATIO"] = str(float(config.get("acm_special_aug_ratio", acm_preset.get("special_aug_ratio", 0.7))))
        acm_ema = as_bool(config.get("acm_ema"), acm_preset.get("ema", False))
        acm_freeze_train = as_bool(config.get("acm_freeze_train"), acm_preset.get("freeze_train", False))
        env["ACM_EMA"] = "1" if acm_ema else "0"
        env["ACM_FREEZE_TRAIN"] = "1" if acm_freeze_train else "0"
        env["ACM_FREEZE_EPOCHS"] = str(int(value_or_default(config.get("acm_freeze_epochs"), acm_preset.get("freeze_epochs", 50))))
        env["ACM_LR_DECAY_TYPE"] = str(config.get("acm_lr_decay_type") or acm_preset.get("lr_decay_type", "cos"))

        eval_flag = as_bool(config.get("acm_eval_flag", config.get("eval_flag", True)), True)
        eval_period = int(config.get("acm_eval_period", config.get("eval_period", 10)))
        eval_json_path = str(config.get("acm_eval_json_path") or default_predict.get("json_path", ""))
        eval_image_root = resolve_dataset_path(
            str(config.get("acm_eval_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        eval_confidence = eval_confidence_global
        eval_nms_iou = eval_nms_iou_global

        env["ACM_EVAL_FLAG"] = "1" if eval_flag else "0"
        env["ACM_EVAL_PERIOD"] = str(eval_period)
        env["ACM_EVAL_JSON_PATH"] = eval_json_path
        env["ACM_EVAL_IMAGE_ROOT"] = eval_image_root
        env["ACM_EVAL_CONFIDENCE"] = str(eval_confidence)
        env["ACM_EVAL_NMS_IOU"] = str(eval_nms_iou)

        print(
            "ACM eval:",
            f"enabled={eval_flag}, period={eval_period}, confidence={eval_confidence}, nms_iou={eval_nms_iou}, json={eval_json_path or '(none)'}, image_root={eval_image_root or '(none)'}",
        )
        print(
            "ACM train:",
            f"mode={acm_det_mode}, backbone={env['ACM_BACKBONE_MODE']}, base_size={env['ACM_BASE_SIZE']}, batch_size={env['ACM_BATCH_SIZE']}, stride={env['ACM_STRIDE']}, mosaic={acm_mosaic}, mixup={acm_mixup}, ema={acm_ema}, freeze={acm_freeze_train}, freeze_epochs={env['ACM_FREEZE_EPOCHS']}, dataset=detlab, loss=detlab-current",
        )

    if is_alc_network(net_key):
        alc_preset = network_presets.get(net_key, network_presets.get("alcnet", {}))
        default_predict = preset.get("default_predict", {})
        alc_det_mode = str(config.get("alc_det_mode", alc_preset.get("det_mode", "")) or "")
        if not alc_det_mode:
            alc_det_mode = "saliency" if is_alc_saliency_network(net_key) else "feature"

        env["ALC_FUSE_MODE"] = str(config.get("alc_fuse_mode", alc_preset.get("fuse_mode", "AsymBi")))
        env["ALC_DET_MODE"] = alc_det_mode
        env["ALC_BLOCKS_PER_LAYER"] = str(int(config.get("alc_blocks_per_layer", alc_preset.get("blocks_per_layer", 4))))
        env["ALC_IMAGE_ROOT"] = resolve_dataset_path(
            str(config.get("alc_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        env["ALC_BASE_SIZE"] = str(int(config.get("alc_base_size", alc_preset.get("base_size", 512))))
        env["ALC_BATCH_SIZE"] = str(int(config.get("alc_batch_size", alc_preset.get("batch_size", 4))))
        env["ALC_EPOCHS"] = str(int(config.get("alc_epochs", alc_preset.get("epochs", 100))))
        env["ALC_WARM_UP_EPOCHS"] = str(int(config.get("alc_warm_up_epochs", alc_preset.get("warm_up_epochs", 0))))
        env["ALC_NUM_CLASSES"] = str(int(config.get("alc_num_classes", alc_preset.get("num_classes", 1))))
        env["ALC_STRIDE"] = str(int(config.get("alc_stride", alc_preset.get("stride", 8))))
        env["ALC_NUM_WORKERS"] = str(int(config.get("alc_num_workers", alc_preset.get("num_workers", 2))))
        env["ALC_SAVE_EVERY"] = str(int(config.get("alc_save_every", alc_preset.get("save_period", 10))))
        env["ALC_RUN_NAME"] = str(config.get("alc_run_name", ""))
        alc_mosaic = as_bool(config.get("alc_mosaic"), alc_preset.get("mosaic", True))
        alc_mixup = as_bool(config.get("alc_mixup"), alc_preset.get("mixup", True))
        env["ALC_MOSAIC"] = "1" if alc_mosaic else "0"
        env["ALC_MIXUP"] = "1" if alc_mixup else "0"
        env["ALC_MOSAIC_PROB"] = str(float(config.get("alc_mosaic_prob", alc_preset.get("mosaic_prob", 0.5))))
        env["ALC_MIXUP_PROB"] = str(float(config.get("alc_mixup_prob", alc_preset.get("mixup_prob", 0.5))))
        env["ALC_SPECIAL_AUG_RATIO"] = str(float(config.get("alc_special_aug_ratio", alc_preset.get("special_aug_ratio", 0.7))))

        eval_flag = bool(config.get("alc_eval_flag", config.get("eval_flag", True)))
        eval_period = int(config.get("alc_eval_period", config.get("eval_period", 10)))
        eval_json_path = str(config.get("alc_eval_json_path") or default_predict.get("json_path", ""))
        eval_image_root = resolve_dataset_path(
            str(config.get("alc_eval_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        eval_confidence = eval_confidence_global
        eval_nms_iou = eval_nms_iou_global

        env["ALC_EVAL_FLAG"] = "1" if eval_flag else "0"
        env["ALC_EVAL_PERIOD"] = str(eval_period)
        env["ALC_EVAL_JSON_PATH"] = eval_json_path
        env["ALC_EVAL_IMAGE_ROOT"] = eval_image_root
        env["ALC_EVAL_CONFIDENCE"] = str(eval_confidence)
        env["ALC_EVAL_NMS_IOU"] = str(eval_nms_iou)

        print(
            "ALC eval:",
            f"enabled={eval_flag}, period={eval_period}, confidence={eval_confidence}, nms_iou={eval_nms_iou}, json={eval_json_path or '(none)'}, image_root={eval_image_root or '(none)'}",
        )
        print(
            "ALC train:",
            f"mode={alc_det_mode}, base_size={env['ALC_BASE_SIZE']}, batch_size={env['ALC_BATCH_SIZE']}, stride={env['ALC_STRIDE']}, dataset=detlab, mosaic={alc_mosaic}, mixup={alc_mixup}, loss=detlab-current",
        )

    if is_sctrans_network(net_key):
        sctrans_preset = network_presets.get(net_key, network_presets.get("sctransnet", {}))
        default_predict = preset.get("default_predict", {})
        env["SCTRANS_IMAGE_ROOT"] = resolve_dataset_path(
            str(config.get("sctrans_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        env["SCTRANS_BASE_SIZE"] = str(int(config.get("sctrans_base_size", sctrans_preset.get("base_size", 512))))
        env["SCTRANS_BATCH_SIZE"] = str(int(config.get("sctrans_batch_size", sctrans_preset.get("batch_size", 4))))
        env["SCTRANS_EPOCHS"] = str(int(config.get("sctrans_epochs", sctrans_preset.get("epochs", 100))))
        env["SCTRANS_WARM_UP_EPOCHS"] = str(int(config.get("sctrans_warm_up_epochs", sctrans_preset.get("warm_up_epochs", 0))))
        env["SCTRANS_NUM_CLASSES"] = str(int(config.get("sctrans_num_classes", sctrans_preset.get("num_classes", 1))))
        env["SCTRANS_STRIDE"] = str(int(config.get("sctrans_stride", sctrans_preset.get("stride", 8))))
        env["SCTRANS_NUM_WORKERS"] = str(int(config.get("sctrans_num_workers", sctrans_preset.get("num_workers", 2))))
        env["SCTRANS_SAVE_EVERY"] = str(int(config.get("sctrans_save_every", sctrans_preset.get("save_period", 10))))
        sctrans_mosaic = as_bool(config.get("sctrans_mosaic"), sctrans_preset.get("mosaic", True))
        sctrans_mixup = as_bool(config.get("sctrans_mixup"), sctrans_preset.get("mixup", True))
        env["SCTRANS_MOSAIC"] = "1" if sctrans_mosaic else "0"
        env["SCTRANS_MIXUP"] = "1" if sctrans_mixup else "0"
        env["SCTRANS_MOSAIC_PROB"] = str(float(config.get("sctrans_mosaic_prob", sctrans_preset.get("mosaic_prob", 0.5))))
        env["SCTRANS_MIXUP_PROB"] = str(float(config.get("sctrans_mixup_prob", sctrans_preset.get("mixup_prob", 0.5))))
        env["SCTRANS_SPECIAL_AUG_RATIO"] = str(float(config.get("sctrans_special_aug_ratio", sctrans_preset.get("special_aug_ratio", 0.7))))

        eval_flag = bool(config.get("sctrans_eval_flag", config.get("eval_flag", True)))
        eval_period = int(config.get("sctrans_eval_period", config.get("eval_period", 10)))
        eval_json_path = str(config.get("sctrans_eval_json_path") or default_predict.get("json_path", ""))
        eval_image_root = resolve_dataset_path(
            str(config.get("sctrans_eval_image_root") or default_predict.get("dataset_img_path", "")),
            dataset_root_prefix,
        )
        eval_confidence = eval_confidence_global
        eval_nms_iou = eval_nms_iou_global

        env["SCTRANS_EVAL_FLAG"] = "1" if eval_flag else "0"
        env["SCTRANS_EVAL_PERIOD"] = str(eval_period)
        env["SCTRANS_EVAL_JSON_PATH"] = eval_json_path
        env["SCTRANS_EVAL_IMAGE_ROOT"] = eval_image_root
        env["SCTRANS_EVAL_CONFIDENCE"] = str(eval_confidence)
        env["SCTRANS_EVAL_NMS_IOU"] = str(eval_nms_iou)

        print(
            "SCTransNet eval:",
            f"enabled={eval_flag}, period={eval_period}, confidence={eval_confidence}, nms_iou={eval_nms_iou}, json={eval_json_path or '(none)'}, image_root={eval_image_root or '(none)'}",
        )
        print(
            "SCTransNet train:",
            f"base_size={env['SCTRANS_BASE_SIZE']}, batch_size={env['SCTRANS_BATCH_SIZE']}, stride={env['SCTRANS_STRIDE']}, dataset=detlab, mosaic={sctrans_mosaic}, mixup={sctrans_mixup}, loss=detlab-current",
        )

    run_command([sys.executable, train_script], env=env, dry_run=dry_run)


def parse_args():
    parser = argparse.ArgumentParser("Train-only experiment runner (JSON config)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to train config json")
    parser.add_argument("--presets", default=DEFAULT_PRESETS_PATH, help="Path to presets json")
    parser.add_argument("--dataset", help="Override dataset from config (highest priority)")
    parser.add_argument("--network", help="Override network from config (highest priority)")
    parser.add_argument("--model-path", help="Override pretrained model path (highest priority)")
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adagrad"], help="Override optimizer (highest priority)")
    parser.add_argument("--init-lr", type=float, help="Override init lr (highest priority)")
    parser.add_argument("--min-lr", type=float, help="Override min lr (highest priority)")
    parser.add_argument("--momentum", type=float, help="Override momentum (highest priority)")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay (highest priority)")
    parser.add_argument("--epochs", type=int, help="Override total epochs for all supported networks (highest priority)")
    parser.add_argument("--eval-period", type=int, help="Override eval period for all supported networks (highest priority)")
    parser.add_argument("--dataset-root-prefix", help="Override dataset root prefix (highest priority)")
    parser.add_argument(
        "--letterbox-train",
        nargs="?",
        const="1",
        help="For networks using utils/acm/data_detlab.py, force train random=False and disable mosaic/mixup. Accepts 1/0.",
    )
    parser.add_argument(
        "--box-coord-type",
        choices=["float", "int"],
        help="For DETLAB txt datasets, parse box coordinates as float by default or int for integer coordinates.",
    )
    parser.add_argument(
        "--box-precision",
        choices=["high", "float", "truncate", "truncated"],
        default="high",
        help="Unified box precision mode. Default high keeps float geometry; truncate uses integer-like geometry.",
    )
    parser.add_argument(
        "--box-geometry-dtype",
        choices=["float", "int"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sequence-box-mode",
        choices=["legacy", "fixed", "float", "float-copy"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved command and output paths")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    presets = load_json(args.presets)
    config = load_json(args.config)

    if args.dataset:
        config["dataset"] = args.dataset
    if args.network:
        config["network"] = args.network
    if args.model_path is not None:
        config["train_model_path"] = args.model_path
    if args.optimizer:
        config["optimizer_type"] = args.optimizer
    if args.init_lr is not None:
        config["init_lr"] = args.init_lr
    if args.min_lr is not None:
        config["min_lr"] = args.min_lr
    if args.momentum is not None:
        config["momentum"] = args.momentum
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.epochs is not None:
        config["epochs"] = args.epochs
        config["acm_epochs"] = args.epochs
        config["alc_epochs"] = args.epochs
        config["dna_epochs"] = args.epochs
        config["uiu_epochs"] = args.epochs
        config["sctrans_epochs"] = args.epochs
        config["dqaligner_epochs"] = args.epochs
    if args.eval_period is not None:
        config["eval_period"] = args.eval_period
        config["acm_eval_period"] = args.eval_period
        config["alc_eval_period"] = args.eval_period
        config["dna_eval_period"] = args.eval_period
        config["uiu_eval_period"] = args.eval_period
        config["sctrans_eval_period"] = args.eval_period
    if args.dataset_root_prefix is not None:
        config["dataset_root_prefix"] = args.dataset_root_prefix
    if args.letterbox_train is not None:
        config["letterbox_train"] = args.letterbox_train
    if args.box_coord_type is not None:
        config["box_coord_type"] = args.box_coord_type
    if args.box_precision is not None:
        config["box_precision"] = args.box_precision
    if args.box_geometry_dtype is not None:
        config["box_geometry_dtype"] = args.box_geometry_dtype
    if args.sequence_box_mode is not None:
        config["sequence_box_mode"] = args.sequence_box_mode
    run_train(config, presets, dry_run=args.dry_run)

