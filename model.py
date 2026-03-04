import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet import random_set

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class RegressionModelDirichlet(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_anchors=9,
        feature_size=256,
        delta_clip=3.0,
    ):
        super(RegressionModelDirichlet, self).__init__()

        self.delta_clip = float(delta_clip)

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 12, kernel_size=3, padding=1)

    def _alphas_to_norm(self, alphas):
        a = alphas.view(-1, 4, 3)
        a_sum = torch.clamp(a.sum(dim=-1, keepdim=True), min=1e-6)
        p_mean = a / a_sum
        p_mean = torch.nan_to_num(p_mean, nan=0.0, posinf=1.0, neginf=0.0)
        bin_centers = torch.tensor([0.1667, 0.5, 0.8333], device=alphas.device)
        coords01 = (p_mean * bin_centers.view(1, 1, 3)).sum(dim=-1)
        return torch.clamp(coords01, 0.0, 1.0)

    def _norm_to_deltas(self, norm):
        return (norm - 0.5) * (2.0 * self.delta_clip)

    def alphas_to_deltas(self, alphas, batch_size=None):
        norm = self._alphas_to_norm(alphas)
        deltas = self._norm_to_deltas(norm)
        if batch_size is not None:
            deltas = deltas.view(batch_size, -1, 4)
        return deltas

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = out.permute(0, 2, 3, 1)

        out = out.contiguous().view(out.shape[0], -1, 12)
        out = F.softplus(out) + 1e-3
        out = torch.clamp(out, 1e-3, 20.0)

        return out


class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_anchors=9,
        num_classes=80,
        prior=0.01,
        feature_size=256,
        random_set_path=None,
        random_set_base_class_names=None,
    ):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.use_random_set = random_set_path is not None

        if self.use_random_set:
            if random_set_base_class_names is None:
                base_class_names = [c["name"] for c in random_set.COCO_CATEGORIES]
            else:
                base_class_names = [str(x) for x in random_set_base_class_names]
            if len(base_class_names) != int(num_classes):
                raise ValueError(
                    "Random-set base class name count ({}) must match num_classes ({}).".format(
                        len(base_class_names), int(num_classes)
                    )
                )
            new_classes = random_set.load_random_set_classes(random_set_path)
            rs_mats = random_set.build_random_set_matrices(new_classes, base_class_names)
            self.register_buffer("rs_membership", rs_mats["membership"])
            self.register_buffer("rs_mass_coeff", rs_mats["mass_coeff"])
            self.register_buffer("rs_pignistic", rs_mats["pignistic"])
            self.random_set_num_classes = len(new_classes)
        else:
            self.random_set_num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchors * self.random_set_num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.random_set_num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.random_set_num_classes)

    def beliefs_to_label_scores(self, beliefs):
        if not self.use_random_set:
            return beliefs
        mass = random_set.belief_to_mass(beliefs, self.rs_mass_coeff, clamp_negative=True)
        betp = random_set.final_betp(mass, self.rs_pignistic)
        return betp


class ResNet(nn.Module):

    def __init__(
        self,
        num_classes,
        block,
        layers,
        use_dirichlet=False,
        use_random_set=False,
        random_set_betp_loss=False,
        random_set_path=None,
        random_set_alpha=0.001,
        random_set_beta=0.001,
        random_set_base_class_names=None,
        dirichlet_coord_l1_weight=1.0,
        dirichlet_kl_weight=0.005,
        dirichlet_delta_clip=3.0,
        dirichlet_target_concentration=20.0,
        score_threshold=0.05,
        nms_iou_threshold=0.5,
        pre_nms_topk=1000,
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.use_dirichlet = bool(use_dirichlet)
        self.use_random_set = bool(use_random_set)
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.pre_nms_topk = int(pre_nms_topk) if pre_nms_topk is not None else None
        if self.use_random_set and not self.use_dirichlet:
            raise ValueError('Random-set classifier is supported only with Dirichlet regression.')
        if self.use_random_set and not random_set_path:
            raise ValueError('random_set_path is required when use_random_set=True.')
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        if self.use_dirichlet:
            self.regressionModel = RegressionModelDirichlet(
                256, delta_clip=dirichlet_delta_clip
            )
        else:
            self.regressionModel = RegressionModel(256)

        self.classificationModel = ClassificationModel(
            256,
            num_classes=num_classes,
            random_set_path=random_set_path if self.use_random_set else None,
            random_set_base_class_names=random_set_base_class_names if self.use_random_set else None,
        )

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        if self.use_dirichlet:
            self.focalLoss = losses.DirichletFocalLoss(
                use_random_set=self.use_random_set,
                use_random_set_betp_loss=random_set_betp_loss,
                random_set_path=random_set_path,
                random_set_alpha=random_set_alpha,
                random_set_beta=random_set_beta,
                random_set_base_class_names=random_set_base_class_names if self.use_random_set else None,
                coord_l1_weight=dirichlet_coord_l1_weight,
                kl_weight=dirichlet_kl_weight,
                delta_clip=dirichlet_delta_clip,
                target_concentration=dirichlet_target_concentration,
            )
        else:
            self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            if self.use_dirichlet:
                regression = self.regressionModel.alphas_to_deltas(regression, img_batch.shape[0])
            if self.use_random_set:
                classification = self.classificationModel.beliefs_to_label_scores(classification)
            if regression.dim() == 2:
                regression = regression.unsqueeze(0)

            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            if classification.dim() == 3:
                if classification.shape[0] != 1:
                    raise ValueError("Inference path expects batch size 1.")
                classification = classification[0]
            if transformed_anchors.dim() == 3:
                if transformed_anchors.shape[0] != 1:
                    raise ValueError("Inference path expects batch size 1.")
                transformed_anchors = transformed_anchors[0]

            scores_kept = []
            labels_kept = []
            boxes_kept = []

            for i in range(classification.shape[1]):
                scores = classification[:, i]
                scores_over_thresh = (scores > self.score_threshold)
                if scores_over_thresh.sum() == 0:
                    continue

                scores = scores[scores_over_thresh]
                anchor_boxes = transformed_anchors[scores_over_thresh]

                # Cap candidates before NMS for much faster eval when scores are dense.
                if self.pre_nms_topk is not None and scores.numel() > self.pre_nms_topk:
                    top_scores, top_idx = torch.topk(
                        scores, k=self.pre_nms_topk, largest=True, sorted=True
                    )
                    scores = top_scores
                    anchor_boxes = anchor_boxes[top_idx]

                anchors_nms_idx = nms(anchor_boxes, scores, self.nms_iou_threshold)
                if anchors_nms_idx.numel() == 0:
                    continue

                cls_scores = scores[anchors_nms_idx]
                cls_boxes = anchor_boxes[anchors_nms_idx]
                cls_labels = torch.full(
                    (anchors_nms_idx.shape[0],),
                    i,
                    dtype=torch.long,
                    device=cls_scores.device,
                )

                scores_kept.append(cls_scores)
                labels_kept.append(cls_labels)
                boxes_kept.append(cls_boxes)

            if not scores_kept:
                empty_scores = classification.new_zeros((0,))
                empty_labels = torch.zeros((0,), dtype=torch.long, device=classification.device)
                empty_boxes = transformed_anchors.new_zeros((0, 4))
                return [empty_scores, empty_labels, empty_boxes]

            finalScores = torch.cat(scores_kept, dim=0)
            finalAnchorBoxesIndexes = torch.cat(labels_kept, dim=0)
            finalAnchorBoxesCoordinates = torch.cat(boxes_kept, dim=0)

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
