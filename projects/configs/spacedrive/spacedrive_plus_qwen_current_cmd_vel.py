_base_ = ['./spacedrive_plus_qwen.py']

custom_imports = dict(
    imports=['projects.mmdet3d_plugin.models.vlm.spacedrive_current_cmd_vel'],
    allow_failed_imports=False,
)

# Keep the plus model recipe, but restrict ego status to current command +
# current velocity only inside the custom detector.
ego_status = 'feature'

base_path = 'workspace/spacedrive_qwen2.5-7b-plus-current-cmd-vel/'
work_dir = base_path
results_path = base_path + '_results_planning_only/'
wb_run_name = base_path + 'debug_run'

model = dict(
    type='SpaceDriveCurrentCommandVelocity',
    save_path=results_path,
    ego_status=ego_status,
)
