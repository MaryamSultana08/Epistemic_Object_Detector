import argparse
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer, collater
from retinanet import coco_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))


def split_minival_ids(image_ids, minival_size, seed):
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    minival_ids = ids[:minival_size]
    trainval_ids = ids[minival_size:]
    return trainval_ids, minival_ids


def build_model(
    depth,
    num_classes,
    use_dirichlet,
    use_random_set,
    args,
    score_threshold_for_model,
    random_set_base_class_names=None,
):
    kwargs = dict(
        num_classes=num_classes,
        pretrained=False,  # IMPORTANT for evaluation
        use_dirichlet=use_dirichlet,
        use_random_set=use_random_set,
        random_set_betp_loss=args.random_set_betp_loss,
        random_set_path=args.random_set_path,
        random_set_alpha=args.random_set_alpha,
        random_set_beta=args.random_set_beta,
        random_set_base_class_names=random_set_base_class_names if use_random_set else None,
        dirichlet_coord_l1_weight=args.dirichlet_coord_l1_weight,
        dirichlet_kl_weight=args.dirichlet_kl_weight,
        dirichlet_delta_clip=args.dirichlet_delta_clip,
        dirichlet_target_concentration=args.dirichlet_target_concentration,
        score_threshold=score_threshold_for_model,  # force low for debugging
        nms_iou_threshold=args.model_nms_iou_threshold,
        pre_nms_topk=args.model_pre_nms_topk,
    )

    if depth == 18:
        return model.resnet18(**kwargs)
    if depth == 34:
        return model.resnet34(**kwargs)
    if depth == 50:
        return model.resnet50(**kwargs)
    if depth == 101:
        return model.resnet101(**kwargs)
    if depth == 152:
        return model.resnet152(**kwargs)
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


def main(args=None):
    parser = argparse.ArgumentParser(description='COCO validation for RetinaNet.')

    parser.add_argument('--coco_path', required=True, help='Path to COCO directory')
    parser.add_argument('--model_path', required=True, help='Path to checkpoint (.pt)', type=str)
    parser.add_argument(
        '--model_variant',
        choices=['baseline', 'dirichlet_std', 'dirichlet_randomset'],
        default='baseline',
    )

    parser.add_argument(
        '--random_set_path',
        default='',
        help='Path to random-set class clusters (required for dirichlet_randomset)',
    )
    parser.add_argument('--random_set_alpha', type=float, default=0.001)
    parser.add_argument('--random_set_beta', type=float, default=0.001)
    parser.add_argument('--random_set_betp_loss', action='store_true')

    parser.add_argument('--dirichlet_kl_weight', type=float, default=0.005)
    parser.add_argument('--dirichlet_coord_l1_weight', type=float, default=1.0)
    parser.add_argument('--dirichlet_delta_clip', type=float, default=3.0)
    parser.add_argument('--dirichlet_target_concentration', type=float, default=20.0)

    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--coco_minival_split', action='store_true')
    parser.add_argument('--minival_size', type=int, default=5000)
    parser.add_argument('--minival_seed', type=int, default=42)

    parser.add_argument('--score_threshold', type=float, default=0.05)
    parser.add_argument('--max_dets', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument(
        '--model_nms_iou_threshold',
        type=float,
        default=0.5,
        help='NMS IoU threshold inside model forward pass',
    )
    parser.add_argument(
        '--model_pre_nms_topk',
        type=int,
        default=1000,
        help='Per-class top-k kept before model-side NMS (smaller is faster)',
    )
    parser.add_argument(
        '--skip_eval_postprocess',
        action='store_true',
        help='Skip evaluator-side NMS/top-k because model already post-processes detections',
    )

    parser.add_argument('--debug_dump_path', default=None)
    parser.add_argument('--debug_topk', type=int, default=50)
    parser.add_argument('--debug_max_images', type=int, default=20)

    args = parser.parse_args(args)

    use_dirichlet = args.model_variant in ('dirichlet_std', 'dirichlet_randomset')
    use_random_set = args.model_variant == 'dirichlet_randomset'
    if use_random_set and not args.random_set_path:
        raise ValueError(
            'Must provide --random_set_path when using --model_variant dirichlet_randomset.'
        )

    # Dataset (IMPORTANT: pass use_dirichlet if your dataset supports it)
    dataset_val = CocoDataset(
        args.coco_path,
        set_name='val2017',
        transform=transforms.Compose([Normalizer(), Resizer()]),
        use_dirichlet=use_dirichlet,  # <-- key change
        annotation_path=None,
    )

    if args.coco_minival_split:
        _, minival_ids = split_minival_ids(dataset_val.image_ids, args.minival_size, args.minival_seed)
        dataset_val.image_ids = minival_ids

    # Keep model-time filtering aligned with eval threshold to avoid extra NMS work.
    model_internal_thresh = args.score_threshold

    random_set_base_class_names = None
    if use_random_set:
        random_set_base_class_names = [str(dataset_val.labels[i]) for i in range(dataset_val.num_classes())]

    net = build_model(
        depth=args.depth,
        num_classes=dataset_val.num_classes(),
        use_dirichlet=use_dirichlet,
        use_random_set=use_random_set,
        args=args,
        score_threshold_for_model=model_internal_thresh,
        random_set_base_class_names=random_set_base_class_names,
    )

    if torch.cuda.is_available():
        net = net.cuda()

    # Load checkpoint safely
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint:", args.model_path)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if len(missing) and len(missing) < 50:
        print("Missing key examples:", missing[:10])
    if len(unexpected) and len(unexpected) < 50:
        print("Unexpected key examples:", unexpected[:10])

    # Wrap after loading
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    net.eval()
    net.module.freeze_bn()

    # DataLoader optional
    data_loader = None
    if args.num_workers and args.num_workers > 0:
        data_loader = DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collater,
        )

    # Run COCO eval
    coco_eval.evaluate_coco(
        dataset_val,
        net,
        threshold=args.score_threshold,
        max_dets=args.max_dets,
        data_loader=data_loader,
        debug_dump_path=args.debug_dump_path,
        debug_topk=args.debug_topk,
        debug_max_images=args.debug_max_images if args.debug_max_images > 0 else None,
        skip_eval_postprocess=args.skip_eval_postprocess,
    )


if __name__ == '__main__':
    main()
