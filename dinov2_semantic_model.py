import torch
import torch.nn as nn
import torch.nn.functional as F


class Dinov2SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(Dinov2SegmentationModel, self).__init__()
        
        # Load DINOv2 backbone
        backbone_name = "dinov2_vits14"
        DINOBackbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        self.backbone = DINOBackbone
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Add a feature pyramid segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)  # Final classification layer
        )
        
        # Global Context Aggregation Branch
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduce to 1x1
            nn.Conv2d(384, 256, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)  # Class logits from global features
        )

    def forward(self, x):
        # Pass input through the DINOv2 backbone
        features = self.backbone(x)  # Output shape: [batch_size, 384]

        # Reshape to [batch_size, 384, 1, 1] for spatial processing
        batch_size = features.size(0)
        features = features.view(batch_size, 384, 1, 1)

        # Expand features spatially
        features = F.interpolate(features, size=(56, 112), mode='bilinear', align_corners=False)

        # Apply segmentation head
        seg_logits = self.seg_head(features)  # Shape: [batch_size, num_classes, 56, 112]

        # Apply global context aggregation
        global_context_logits = self.global_context(features)  # Shape: [batch_size, num_classes, 1, 1]
        global_context_logits = F.interpolate(global_context_logits, size=(560, 1120), mode='bilinear', align_corners=False)

        # Upsample seg_logits to match global_context_logits
        seg_logits = F.interpolate(seg_logits, size=(560, 1120), mode='bilinear', align_corners=False)

        # Combine the outputs
        final_logits = seg_logits + global_context_logits

        return final_logits

# import torch
# from transformers import Dinov2Model, Dinov2PreTrainedModel
# from transformers.modeling_outputs import SemanticSegmenterOutput


# class LinearClassifier(torch.nn.Module):
#     def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=19):  # Update num_labels for Cityscapes
#         super(LinearClassifier, self).__init__()

#         self.in_channels = in_channels
#         self.width = tokenW
#         self.height = tokenH
#         self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

#     def forward(self, embeddings):
#         # Reshape patch embeddings to 2D spatial format
#         embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
#         embeddings = embeddings.permute(0, 3, 1, 2)  # [B, C, H, W]
#         return self.classifier(embeddings)


# class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         # Initialize DINOv2 model
#         self.dinov2 = Dinov2Model(config)
#         self.classifier = LinearClassifier(config.hidden_size, 32, 32, num_labels=config.num_labels)

#         # Optional: Unfreeze backbone for fine-tuning
#         for param in self.dinov2.parameters():
#             param.requires_grad = True

#     def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
#         # Extract patch embeddings from DINOv2
#         outputs = self.dinov2(pixel_values,
#                               output_hidden_states=output_hidden_states,
#                               output_attentions=output_attentions)

#         # Exclude CLS token
#         patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Shape: [B, tokens, hidden_size]

#         # Convert to logits and upsample to match pixel_values' resolution
#         logits = self.classifier(patch_embeddings)
#         logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

#         # Compute loss
#         loss = None
#         if labels is not None:
#             # Update ignore_index for Cityscapes
#             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
#             loss = loss_fct(logits, labels)

#         return SemanticSegmenterOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
