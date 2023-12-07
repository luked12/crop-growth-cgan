import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes, version='basic'):
    # load an instance segmentation model pre-trained on COCO
    if version == 'basic':
        # v1: standard ResNet50 backbone
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.DEFAULT')
    elif version == 'transformer':
        # v2 Improved Mask R-CNN model with a ResNet-50-FPN backbone from the Benchmarking Detection Transfer Learning with Vision Transformers paper.
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT') # v2: Improved Mask R-CNN model with a ResNet-50-FPN backbone from the Benchmarking Detection Transfer Learning with Vision Transformers paper.

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model