_base_ = '../configs/spacedrive/spacedrive_plus_qwen.py'

custom_imports = dict(
    imports=['projects.KLAL.spacedrive_klal'],
    allow_failed_imports=False,
)

# KLAL config.  Last 4 layers are a safer default than all layers because
# output_attentions=True stores (B, heads, seq, seq) tensors.
klal_lambda = 1.0
klal_layers = [-4, -3, -2, -1]
klal_gt_dir = 'gt_annotation/klal_gt_attention_maps_train_all'

model = dict(
    type='SpaceDriveKLAL',
    use_klal=True,
    klal_gt_dir=klal_gt_dir,
    klal_lambda=klal_lambda,
    klal_layers=klal_layers,
)
