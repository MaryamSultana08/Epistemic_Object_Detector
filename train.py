import argparse
import copy
import collections
import datetime
import errno
import os
import random
import shutil

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

if torch.__version__.split('.')[0] != '1':
    print('Warning: expected PyTorch 1.x, running {}'.format(torch.__version__))


class Logger:
    def __init__(self, path):
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.file = open(path, 'a')
        self.disabled = False

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{timestamp}] {msg}'
        print(line)
        if self.disabled:
            return
        try:
            self.file.write(line + '\n')
            self.file.flush()
        except OSError as e:
            # Avoid crashing training when disk is full or logging fails.
            if e.errno == errno.ENOSPC:
                self.disabled = True
                try:
                    self.file.close()
                except OSError:
                    pass
                return
            raise

    def close(self):
        if self.file:
            self.file.close()


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.cum_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum_sizes.append(total)

    def __len__(self):
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx):
        for ds_idx, end in enumerate(self.cum_sizes):
            if idx < end:
                start = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
                return self.datasets[ds_idx][idx - start]
        raise IndexError('Index out of range')

    def image_aspect_ratio(self, idx):
        for ds_idx, end in enumerate(self.cum_sizes):
            if idx < end:
                start = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
                return self.datasets[ds_idx].image_aspect_ratio(idx - start)
        raise IndexError('Index out of range')


def split_minival_ids(image_ids, minival_size, seed):
    ids = list(image_ids)
    random.Random(seed).shuffle(ids)
    minival_ids = set(ids[:minival_size])
    trainval_ids = [i for i in ids if i not in minival_ids]
    return trainval_ids, list(minival_ids)


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', choices=['csv', 'coco'], required=True,
                        help='Dataset type')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--log_path', help='Path to training log file', default='retinanet_training.log')
    parser.add_argument('--checkpoint_dir', default='checkpoints',
                        help='Directory to save per-epoch checkpoints')
    parser.add_argument('--checkpoint_last', default='last.pt',
                        help='Path for the last checkpoint (overwritten each epoch)')
    parser.add_argument('--save_epoch_checkpoints', action='store_true',
                        help='Save a per-epoch checkpoint in checkpoint_dir')
    parser.add_argument('--checkpoint_min_free_mb', type=float, default=128.0,
                        help='Skip checkpoint save when free disk would drop below this threshold (MB)')
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='Disable ImageNet pretrained backbone initialization')
    parser.add_argument('--resume_checkpoint', default='',
                        help='Path to a checkpoint to resume training from')
    parser.add_argument('--loss_csv_path', default='',
                        help='Optional CSV path to save per-iteration losses')
    parser.add_argument(
        '--model_variant',
        choices=['baseline', 'dirichlet_std', 'dirichlet_randomset'],
        default='baseline',
        help='Choose baseline, Dirichlet+standard classifier, or Dirichlet+random-set classifier',
    )
    parser.add_argument('--random_set_path', default='',
                        help='Path to random-set class clusters (required for dirichlet_randomset)')
    parser.add_argument('--random_set_alpha', type=float, default=0.001,
                        help='Random-set negative-mass regularizer weight')
    parser.add_argument('--random_set_beta', type=float, default=0.001,
                        help='Random-set mass-sum regularizer weight')
    parser.add_argument('--random_set_betp_loss', action='store_true',
                        help='Use BetP + standard focal BCE loss for random-set classifier')
    parser.add_argument('--dirichlet_kl_weight', type=float, default=0.005,
                        help='Dirichlet KL loss weight')
    parser.add_argument('--dirichlet_coord_l1_weight', type=float, default=1.0,
                        help='Dirichlet coordinate L1 loss weight')
    parser.add_argument('--dirichlet_delta_clip', type=float, default=3.0,
                        help='Dirichlet delta clip for regression targets')
    parser.add_argument('--dirichlet_target_concentration', type=float, default=20.0,
                        help='Dirichlet target concentration for bin interpolation')

    parser.add_argument('--coco_minival_split', action='store_true',
                        help='Use train2017 + trainval35k for training and minival for validation')
    parser.add_argument('--minival_size', type=int, default=5000,
                        help='Number of images to use for minival split (default: 5000)')
    parser.add_argument('--minival_seed', type=int, default=42,
                        help='Random seed used for minival split')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--max_train_images', type=int, default=0,
                        help='Limit number of training images (0 = all)')
    parser.add_argument('--max_val_images', type=int, default=0,
                        help='Limit number of validation images (0 = all)')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Run detection evaluation every N epochs (0 disables)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip COCO/CSV detection evaluation during training')
    parser.add_argument('--eval_max_images', type=int, default=0,
                        help='Limit validation images used for detection evaluation (0 = all)')
    parser.add_argument('--eval_score_threshold', type=float, default=0.05,
                        help='Score threshold for COCO detection evaluation')
    parser.add_argument('--eval_max_dets', type=int, default=100,
                        help='Max detections per image for COCO detection evaluation')
    parser.add_argument('--eval_pre_nms_topk', type=int, default=1000,
                        help='Evaluator pre-NMS top-k (ignored with --eval_skip_postprocess)')
    parser.add_argument('--eval_nms_iou_threshold', type=float, default=0.5,
                        help='Evaluator NMS IoU threshold (ignored with --eval_skip_postprocess)')
    parser.add_argument('--eval_skip_postprocess', action='store_true',
                        help='Skip evaluator-side NMS/top-k; use model post-processed detections directly')
    parser.add_argument('--eval_write_results_json', action='store_true',
                        help='Write COCO eval detections to <set>_bbox_results.json (default: disabled)')
    parser.add_argument('--eval_debug_max_images', type=int, default=0,
                        help='Debug cap passed to COCO evaluator (0 = disabled)')
    parser.add_argument('--skip_val_loss', action='store_true',
                        help='Skip validation-loss computation at epoch end')
    parser.add_argument('--val_loss_max_images', type=int, default=0,
                        help='Limit images used for validation-loss computation (0 = all)')
    parser.add_argument('--save_before_eval', action='store_true',
                        help='Save checkpoint at end of epoch before running evaluation')
    parser.add_argument('--model_pre_nms_topk', type=int, default=1000,
                        help='Model-side per-class top-k before NMS during inference')
    parser.add_argument('--model_nms_iou_threshold', type=float, default=0.5,
                        help='Model-side NMS IoU threshold during inference')

    parser = parser.parse_args(args)

    logger = Logger(parser.log_path)
    logger.log('CUDA available: {}'.format(torch.cuda.is_available()))
    logger.log('Model variant: {}'.format(parser.model_variant))
    logger.log(
        'Loss weights: focal(alpha=0.25,gamma=2.0), '
        f'dirichlet(kl={parser.dirichlet_kl_weight}, '
        f'coord_l1={parser.dirichlet_coord_l1_weight}, '
        f'delta_clip={parser.dirichlet_delta_clip}, '
        f'target_concentration={parser.dirichlet_target_concentration}), '
        f'random_set(alpha={parser.random_set_alpha}, beta={parser.random_set_beta}, '
        f'betp_loss={parser.random_set_betp_loss})'
    )
    logger.log(
        'Eval config: '
        f'skip_eval={parser.skip_eval}, '
        f'eval_every={parser.eval_every}, '
        f'eval_max_images={parser.eval_max_images}, '
        f'skip_val_loss={parser.skip_val_loss}, '
        f'val_loss_max_images={parser.val_loss_max_images}, '
        f'eval_skip_postprocess={parser.eval_skip_postprocess}, '
        f'eval_write_results_json={parser.eval_write_results_json}, '
        f'model_pre_nms_topk={parser.model_pre_nms_topk}'
    )
    pretrained_backbone = (not parser.no_pretrained_backbone) and (not parser.resume_checkpoint)
    if parser.resume_checkpoint and not parser.no_pretrained_backbone:
        logger.log('Resume checkpoint provided; disabling ImageNet pretrained backbone init')
    elif parser.no_pretrained_backbone:
        logger.log('ImageNet pretrained backbone init disabled by --no_pretrained_backbone')
    else:
        logger.log('ImageNet pretrained backbone init enabled')

    use_dirichlet = parser.model_variant in ('dirichlet_std', 'dirichlet_randomset')
    use_random_set = parser.model_variant == 'dirichlet_randomset'
    if use_random_set and not parser.random_set_path:
        raise ValueError(
            'Must provide --random_set_path when using --model_variant dirichlet_randomset.'
        )

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        train_transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        val_transform = transforms.Compose([Normalizer(), Resizer()])

        dataset_train_2017 = CocoDataset(
            parser.coco_path,
            set_name='train2017',
            transform=train_transform,
            use_dirichlet=False,
            annotation_path=None,
        )
        dataset_val_full = CocoDataset(
            parser.coco_path,
            set_name='val2017',
            transform=val_transform,
            use_dirichlet=False,
            annotation_path=None,
        )

        if parser.coco_minival_split:
            trainval_ids, minival_ids = split_minival_ids(
                dataset_val_full.image_ids, parser.minival_size, parser.minival_seed
            )
            dataset_trainval = CocoDataset(
                parser.coco_path,
                set_name='val2017',
                transform=train_transform,
                use_dirichlet=False,
                annotation_path=None,
            )
            dataset_trainval.image_ids = trainval_ids
            dataset_val_full.image_ids = minival_ids
            dataset_train = CombinedDataset([dataset_train_2017, dataset_trainval])
            dataset_val = dataset_val_full
            logger.log('Using train2017 + trainval35k / minival split '
                       f'(trainval={len(trainval_ids)}, minival={len(minival_ids)})')
        else:
            dataset_train = dataset_train_2017
            dataset_val = dataset_val_full
            logger.log('Using train2017 / val2017 split')

    elif parser.dataset == 'csv':

        if use_dirichlet:
            raise ValueError('Dirichlet variants are supported only for COCO in this script.')

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on CSV.')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on CSV.')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            logger.log('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    if parser.max_train_images and parser.max_train_images > 0:
        dataset_train.image_ids = dataset_train.image_ids[:parser.max_train_images]
        logger.log('Limiting train images to {}'.format(len(dataset_train.image_ids)))

    if dataset_val is not None and parser.max_val_images and parser.max_val_images > 0:
        if hasattr(dataset_val, 'image_ids'):
            dataset_val.image_ids = dataset_val.image_ids[:parser.max_val_images]
            logger.log('Limiting val images to {}'.format(len(dataset_val.image_ids)))

    random_set_base_class_names = None
    if use_random_set:
        def _extract_label_names(ds):
            if hasattr(ds, 'labels') and callable(getattr(ds, 'num_classes', None)):
                n = int(ds.num_classes())
                names = [str(ds.labels[i]) for i in range(n)]
                return names
            if hasattr(ds, 'datasets') and isinstance(ds.datasets, list) and ds.datasets:
                return _extract_label_names(ds.datasets[0])
            return None

        random_set_base_class_names = _extract_label_names(dataset_train)
        if random_set_base_class_names is None:
            raise ValueError('Could not derive class names for random-set classifier.')
        logger.log(
            'Random-set base classes ({}): {}'.format(
                len(random_set_base_class_names), random_set_base_class_names
            )
        )

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    dataset_val_for_loss = dataset_val
    if dataset_val is not None and parser.val_loss_max_images and parser.val_loss_max_images > 0:
        if hasattr(dataset_val, 'image_ids'):
            dataset_val_for_loss = copy.copy(dataset_val)
            dataset_val_for_loss.image_ids = dataset_val.image_ids[:parser.val_loss_max_images]
            logger.log('Limiting validation-loss images to {}'.format(len(dataset_val_for_loss.image_ids)))

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val_for_loss, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val_for_loss, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(
            num_classes=dataset_train.num_classes(),
            pretrained=pretrained_backbone,
            use_dirichlet=use_dirichlet,
            use_random_set=use_random_set,
            random_set_betp_loss=parser.random_set_betp_loss,
            random_set_path=parser.random_set_path,
            random_set_alpha=parser.random_set_alpha,
            random_set_beta=parser.random_set_beta,
            random_set_base_class_names=random_set_base_class_names,
            dirichlet_coord_l1_weight=parser.dirichlet_coord_l1_weight,
            dirichlet_kl_weight=parser.dirichlet_kl_weight,
            dirichlet_delta_clip=parser.dirichlet_delta_clip,
            dirichlet_target_concentration=parser.dirichlet_target_concentration,
            nms_iou_threshold=parser.model_nms_iou_threshold,
            pre_nms_topk=parser.model_pre_nms_topk,
        )
    elif parser.depth == 34:
        retinanet = model.resnet34(
            num_classes=dataset_train.num_classes(),
            pretrained=pretrained_backbone,
            use_dirichlet=use_dirichlet,
            use_random_set=use_random_set,
            random_set_betp_loss=parser.random_set_betp_loss,
            random_set_path=parser.random_set_path,
            random_set_alpha=parser.random_set_alpha,
            random_set_beta=parser.random_set_beta,
            random_set_base_class_names=random_set_base_class_names,
            dirichlet_coord_l1_weight=parser.dirichlet_coord_l1_weight,
            dirichlet_kl_weight=parser.dirichlet_kl_weight,
            dirichlet_delta_clip=parser.dirichlet_delta_clip,
            dirichlet_target_concentration=parser.dirichlet_target_concentration,
            nms_iou_threshold=parser.model_nms_iou_threshold,
            pre_nms_topk=parser.model_pre_nms_topk,
        )
    elif parser.depth == 50:
        retinanet = model.resnet50(
            num_classes=dataset_train.num_classes(),
            pretrained=pretrained_backbone,
            use_dirichlet=use_dirichlet,
            use_random_set=use_random_set,
            random_set_betp_loss=parser.random_set_betp_loss,
            random_set_path=parser.random_set_path,
            random_set_alpha=parser.random_set_alpha,
            random_set_beta=parser.random_set_beta,
            random_set_base_class_names=random_set_base_class_names,
            dirichlet_coord_l1_weight=parser.dirichlet_coord_l1_weight,
            dirichlet_kl_weight=parser.dirichlet_kl_weight,
            dirichlet_delta_clip=parser.dirichlet_delta_clip,
            dirichlet_target_concentration=parser.dirichlet_target_concentration,
            nms_iou_threshold=parser.model_nms_iou_threshold,
            pre_nms_topk=parser.model_pre_nms_topk,
        )
    elif parser.depth == 101:
        retinanet = model.resnet101(
            num_classes=dataset_train.num_classes(),
            pretrained=pretrained_backbone,
            use_dirichlet=use_dirichlet,
            use_random_set=use_random_set,
            random_set_betp_loss=parser.random_set_betp_loss,
            random_set_path=parser.random_set_path,
            random_set_alpha=parser.random_set_alpha,
            random_set_beta=parser.random_set_beta,
            random_set_base_class_names=random_set_base_class_names,
            dirichlet_coord_l1_weight=parser.dirichlet_coord_l1_weight,
            dirichlet_kl_weight=parser.dirichlet_kl_weight,
            dirichlet_delta_clip=parser.dirichlet_delta_clip,
            dirichlet_target_concentration=parser.dirichlet_target_concentration,
            nms_iou_threshold=parser.model_nms_iou_threshold,
            pre_nms_topk=parser.model_pre_nms_topk,
        )
    elif parser.depth == 152:
        retinanet = model.resnet152(
            num_classes=dataset_train.num_classes(),
            pretrained=pretrained_backbone,
            use_dirichlet=use_dirichlet,
            use_random_set=use_random_set,
            random_set_betp_loss=parser.random_set_betp_loss,
            random_set_path=parser.random_set_path,
            random_set_alpha=parser.random_set_alpha,
            random_set_beta=parser.random_set_beta,
            random_set_base_class_names=random_set_base_class_names,
            dirichlet_coord_l1_weight=parser.dirichlet_coord_l1_weight,
            dirichlet_kl_weight=parser.dirichlet_kl_weight,
            dirichlet_delta_clip=parser.dirichlet_delta_clip,
            dirichlet_target_concentration=parser.dirichlet_target_concentration,
            nms_iou_threshold=parser.model_nms_iou_threshold,
            pre_nms_topk=parser.model_pre_nms_topk,
        )
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    loss_hist = collections.deque(maxlen=500)

    start_epoch = 0
    if parser.resume_checkpoint:
        logger.log(f'Resuming from checkpoint: {parser.resume_checkpoint}')
        checkpoint = torch.load(parser.resume_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            retinanet.module.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = int(checkpoint.get('epoch', -1)) + 1
        else:
            if hasattr(checkpoint, 'state_dict'):
                retinanet.module.load_state_dict(checkpoint.state_dict())
            else:
                retinanet.module.load_state_dict(checkpoint)

    os.makedirs(parser.checkpoint_dir, exist_ok=True)
    loss_csv_file = None
    if parser.loss_csv_path:
        loss_dir = os.path.dirname(parser.loss_csv_path)
        if loss_dir:
            os.makedirs(loss_dir, exist_ok=True)
        try:
            loss_csv_file = open(parser.loss_csv_path, 'w')
            loss_csv_file.write('epoch,iteration,total,classification,regression,kl_loss,coord_l1_loss\n')
            loss_csv_file.flush()
        except OSError as error:
            if error.errno == errno.ENOSPC:
                logger.log(
                    'Disabling loss CSV logging (disk full): {} ({})'.format(
                        parser.loss_csv_path,
                        error,
                    )
                )
                loss_csv_file = None
            else:
                raise

    retinanet.train()
    retinanet.module.freeze_bn()

    logger.log('Num training images: {}'.format(len(dataset_train)))

    def compute_validation_loss(dataloader):
        retinanet.train()
        retinanet.module.freeze_bn()
        val_loss = []
        with torch.no_grad():
            for data in dataloader:
                if torch.cuda.is_available():
                    outputs = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    outputs = retinanet([data['img'].float(), data['annot']])

                if isinstance(outputs, (list, tuple)) and len(outputs) == 4:
                    classification_loss, regression_loss, kl_loss, l1_loss = outputs
                else:
                    classification_loss, regression_loss = outputs
                    kl_loss = None
                    l1_loss = None

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                if kl_loss is not None:
                    kl_loss = kl_loss.mean()
                if l1_loss is not None:
                    l1_loss = l1_loss.mean()

                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                val_loss.append(float(loss))
        if not val_loss:
            return 0.0
        return float(np.mean(val_loss))

    def save_checkpoint(epoch_num):
        model_to_save = retinanet.module if hasattr(retinanet, 'module') else retinanet
        checkpoint_state = {
            'epoch': epoch_num,
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

        def estimate_serialized_bytes(value):
            total = 0
            stack = [value]
            while stack:
                item = stack.pop()
                if torch.is_tensor(item):
                    total += item.numel() * item.element_size()
                elif isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
            return total

        def is_probable_disk_full(error):
            if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                return True
            message = str(error).lower()
            disk_full_markers = [
                'no space left on device',
                'pytorchstreamwriter failed writing file',
                'file write failed',
                'unexpected pos',
            ]
            return any(marker in message for marker in disk_full_markers)

        def safe_torch_save(state, target_path, label):
            target_dir = os.path.dirname(target_path) or '.'
            os.makedirs(target_dir, exist_ok=True)

            estimated_bytes = estimate_serialized_bytes(state)
            free_bytes = shutil.disk_usage(target_dir).free
            reserve_bytes = int(parser.checkpoint_min_free_mb * 1024 * 1024)
            if free_bytes < estimated_bytes + reserve_bytes:
                logger.log(
                    'Skipping {} save: need ~{:.1f} MB (+{:.1f} MB reserve), only {:.1f} MB free at {}'.format(
                        label,
                        estimated_bytes / (1024 * 1024),
                        parser.checkpoint_min_free_mb,
                        free_bytes / (1024 * 1024),
                        os.path.abspath(target_dir),
                    )
                )
                return False

            temp_path = '{}.tmp'.format(target_path)
            try:
                torch.save(state, temp_path)
                os.replace(temp_path, target_path)
                return True
            except Exception as error:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                if is_probable_disk_full(error):
                    free_after = shutil.disk_usage(target_dir).free
                    logger.log(
                        'Checkpoint save failed (disk full) for {} at {}: {} | free space {:.1f} MB'.format(
                            label,
                            target_path,
                            error,
                            free_after / (1024 * 1024),
                        )
                    )
                    return False
                raise

        last_saved = safe_torch_save(checkpoint_state, parser.checkpoint_last, 'last checkpoint')

        epoch_saved = True
        if parser.save_epoch_checkpoints:
            epoch_name = '{}_{}_epoch{}.pt'.format(parser.dataset, parser.model_variant, epoch_num)
            epoch_path = os.path.join(parser.checkpoint_dir, epoch_name)
            epoch_saved = safe_torch_save(checkpoint_state, epoch_path, 'epoch checkpoint')

        return last_saved and epoch_saved

    def get_eval_dataset():
        if dataset_val is None:
            return None
        if parser.eval_max_images and parser.eval_max_images > 0 and hasattr(dataset_val, 'image_ids'):
            eval_dataset = copy.copy(dataset_val)
            eval_dataset.image_ids = dataset_val.image_ids[:parser.eval_max_images]
            return eval_dataset
        return dataset_val

    for epoch_num in range(start_epoch, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    outputs = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    outputs = retinanet([data['img'].float(), data['annot']])

                if isinstance(outputs, (list, tuple)) and len(outputs) == 4:
                    classification_loss, regression_loss, kl_loss, l1_loss = outputs
                else:
                    classification_loss, regression_loss = outputs
                    kl_loss = None
                    l1_loss = None
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                if kl_loss is not None:
                    kl_loss = kl_loss.mean()
                if l1_loss is not None:
                    l1_loss = l1_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue
                kl_value = float(kl_loss.detach().mean().item()) if kl_loss is not None else 0.0
                l1_value = float(l1_loss.detach().mean().item()) if l1_loss is not None else 0.0

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if use_dirichlet:
                    logger.log(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | KL: {:1.5f} | L1: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num,
                            iter_num,
                            float(classification_loss),
                            float(regression_loss),
                            kl_value,
                            l1_value,
                            np.mean(loss_hist),
                        )
                    )
                else:
                    logger.log(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                if loss_csv_file is not None:
                    try:
                        loss_csv_file.write(
                            '{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                                epoch_num,
                                iter_num,
                                float(loss),
                                float(classification_loss),
                                float(regression_loss),
                                kl_value,
                                l1_value,
                            )
                        )
                        loss_csv_file.flush()
                    except OSError as error:
                        if error.errno == errno.ENOSPC:
                            logger.log('Disabling loss CSV logging (disk full): {}'.format(error))
                            try:
                                loss_csv_file.close()
                            except OSError:
                                pass
                            loss_csv_file = None
                        else:
                            raise

                del classification_loss
                del regression_loss
            except Exception as e:
                logger.log(str(e))
                continue

        if parser.save_before_eval:
            saved_before_eval = save_checkpoint(epoch_num)
            if saved_before_eval:
                logger.log('Saved checkpoint before evaluation: {}'.format(parser.checkpoint_last))
            else:
                logger.log('Skipped checkpoint before evaluation due to checkpoint save conditions')

        run_detection_eval = (
            (not parser.skip_eval)
            and parser.eval_every > 0
            and (epoch_num % parser.eval_every == 0)
        )

        if run_detection_eval and parser.dataset == 'coco' and dataset_val is not None:
            eval_dataset = get_eval_dataset()
            if eval_dataset is not dataset_val and hasattr(eval_dataset, 'image_ids') and hasattr(dataset_val, 'image_ids'):
                logger.log(
                    'Evaluating dataset on {} / {} val images'.format(
                        len(eval_dataset.image_ids), len(dataset_val.image_ids)
                    )
                )
            else:
                logger.log('Evaluating dataset')

            coco_eval.evaluate_coco(
                eval_dataset,
                retinanet,
                threshold=parser.eval_score_threshold,
                write_results=parser.eval_write_results_json,
                max_dets=parser.eval_max_dets,
                nms_iou_threshold=parser.eval_nms_iou_threshold,
                pre_nms_topk=parser.eval_pre_nms_topk,
                debug_max_images=parser.eval_debug_max_images if parser.eval_debug_max_images > 0 else None,
                skip_eval_postprocess=parser.eval_skip_postprocess,
            )
        elif run_detection_eval and parser.dataset == 'csv' and parser.csv_val is not None:
            logger.log('Evaluating dataset')
            mAP = csv_eval.evaluate(dataset_val, retinanet)
            if isinstance(mAP, dict):
                ap_values = [val[0] for val in mAP.values() if isinstance(val, (tuple, list))]
                mean_ap = float(np.mean(ap_values)) if ap_values else 0.0
                logger.log('mAP (mean over classes): {:1.5f}'.format(mean_ap))
            else:
                logger.log('mAP: {:1.5f}'.format(mAP))
        else:
            if parser.skip_eval:
                logger.log('Skipping detection evaluation (--skip_eval)')
            elif parser.eval_every == 0:
                logger.log('Skipping detection evaluation (eval_every=0)')
            else:
                logger.log('Skipping detection evaluation this epoch (eval_every={})'.format(parser.eval_every))

        if dataset_val is not None and not parser.skip_val_loss:
            val_loss = compute_validation_loss(dataloader_val)
            logger.log('Epoch: {} | Validation loss: {:1.5f}'.format(epoch_num, val_loss))
        elif dataset_val is not None and parser.skip_val_loss:
            logger.log('Skipping validation loss (--skip_val_loss)')

        epoch_mean_loss = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        scheduler.step(epoch_mean_loss)
        end_epoch_saved = save_checkpoint(epoch_num)
        if not end_epoch_saved:
            logger.log('Skipped end-of-epoch checkpoint save')

    retinanet.eval()

    final_name = 'model_final_{}.pt'.format(parser.model_variant)
    try:
        torch.save(retinanet, final_name)
    except Exception as error:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            logger.log('Final model save failed (disk full): {}'.format(error))
        else:
            message = str(error).lower()
            if ('no space left on device' in message
                    or 'pytorchstreamwriter failed writing file' in message
                    or 'file write failed' in message
                    or 'unexpected pos' in message):
                logger.log('Final model save failed (disk full): {}'.format(error))
            else:
                raise
    logger.close()
    if loss_csv_file is not None:
        loss_csv_file.close()


if __name__ == '__main__':
    main()
