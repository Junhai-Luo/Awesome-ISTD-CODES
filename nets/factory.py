def build_acm_detector(backbone_mode="FPN", fuse_mode="AsymBi", blocks_per_layer=4, num_classes=1, det_mode="feature"):
    from .acm.detection import ASKCResNetFPNDet, ASKCResUNetDet, ASKCResUNetSaliencyDet

    layer_blocks = [int(blocks_per_layer)] * 3
    channels = [8, 16, 32, 64]
    det_mode = str(det_mode or "feature").lower()
    if str(backbone_mode).lower() == "fpn":
        if det_mode == "saliency":
            raise ValueError("ACM saliency detection is only supported with UNet backbone.")
        return ASKCResNetFPNDet(layer_blocks, channels, fuse_mode, num_classes)
    if str(backbone_mode).lower() in ("unet", "u_net"):
        if det_mode == "saliency":
            return ASKCResUNetSaliencyDet(layer_blocks, channels, fuse_mode, num_classes)
        return ASKCResUNetDet(layer_blocks, channels, fuse_mode, num_classes)
    raise ValueError(f"Unsupported ACM backbone_mode '{backbone_mode}'. Use FPN or UNet.")


def build_alc_detector(fuse_mode="AsymBi", blocks_per_layer=4, num_classes=1, det_mode="feature"):
    from .alc.detection import ALCNetDet, ALCNetSaliencyDet

    layer_blocks = [int(blocks_per_layer)] * 3
    channels = [8, 16, 32, 64]
    model_cls = ALCNetSaliencyDet if str(det_mode or "feature").lower() == "saliency" else ALCNetDet
    return model_cls(in_channels=3, layers=layer_blocks, channels=channels, fuse_mode=fuse_mode, num_classes=num_classes)


def build_sctransnet_detector(num_classes=1, img_size=512):
    from .sctransnet.Config import get_SCTrans_config
    from .sctransnet.SCTransNetDet import SCTransNetDet

    return SCTransNetDet(get_SCTrans_config(), n_channels=3, num_classes=num_classes, img_size=img_size)


def build_dqaligner_detector(num_classes=1, num_frame=5, det_mode="feature"):
    from .dqaligner.detection import DQAlignerDet, DQAlignerSaliencyDet

    model_cls = DQAlignerSaliencyDet if str(det_mode or "feature").lower() == "saliency" else DQAlignerDet
    return model_cls(input_channels=3, num_frames=num_frame, num_classes=num_classes, key_mode="last")


def build_network(network_name, num_classes, num_frame=5):
    name = (network_name or "sstnet").lower()
    if name == "sstnet":
        from .Network import Network

        return Network(num_classes=num_classes, num_frame=num_frame)
    if name in ("tridos", "slowfastnet", "slowfastnet_9520"):
        from .slowfastnet_9520 import slowfastnet

        return slowfastnet(num_classes=num_classes, num_frame=num_frame)
    if name == "dnanet":
        from .dna.model_DNANet_det import DNANetDet

        return DNANetDet(input_channels=3, num_classes=num_classes)
    if name == "dnanet_saliency":
        from .dna.model_DNANet_det import DNANetSaliencyDet

        return DNANetSaliencyDet(input_channels=3, num_classes=num_classes)
    if name == "uiunet":
        from .uiu.detection import UIUNETDet

        return UIUNETDet(in_ch=3, num_classes=num_classes, fuse_mode="AsymBi")
    if name == "uiunet_saliency":
        from .uiu.detection import UIUNETSaliencyDet

        return UIUNETSaliencyDet(in_ch=3, num_classes=num_classes, fuse_mode="AsymBi")
    if name == "acm_fpn":
        return build_acm_detector("FPN", num_classes=num_classes)
    if name == "acm_unet":
        return build_acm_detector("UNet", num_classes=num_classes)
    if name == "acm_unet_saliency":
        return build_acm_detector("UNet", num_classes=num_classes, det_mode="saliency")
    if name == "alcnet":
        return build_alc_detector(num_classes=num_classes)
    if name == "alcnet_saliency":
        return build_alc_detector(num_classes=num_classes, det_mode="saliency")
    if name in ("sctransnet", "sctransnet_det"):
        return build_sctransnet_detector(num_classes=num_classes)
    if name in ("dqaligner", "dqaligner_det"):
        return build_dqaligner_detector(num_classes=num_classes, num_frame=num_frame, det_mode="feature")
    if name in ("dqaligner_saliency", "dqaligner_saliency_det"):
        return build_dqaligner_detector(num_classes=num_classes, num_frame=num_frame, det_mode="saliency")

    raise ValueError(
        f"Unsupported network '{network_name}'. Use one of: sstnet, tridos, slowfastnet, slowfastnet_9520, dnanet, dnanet_saliency, uiunet, uiunet_saliency, acm_fpn, acm_unet, acm_unet_saliency, alcnet, alcnet_saliency, sctransnet, dqaligner, dqaligner_saliency"
    )
