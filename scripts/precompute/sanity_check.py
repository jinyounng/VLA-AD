import pickle
import numpy as np
from pathlib import Path

pkl_path = Path("/data/jykim/projects/SpaceDrive/data/nuscenes/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
infos = data['infos']

# 1. CAN bus 구조 확인
vals = np.array([info['can_bus'] for info in infos[:500]])
print("can_bus shape:", vals.shape)
for i in range(vals.shape[1]):
    col = vals[:, i]
    print(f"  idx {i}: min={col.min():.3f}, max={col.max():.3f}, "
          f"mean={col.mean():.3f}, neg_ratio={(col<0).mean():.2f}")

# 2. gt_planning_command format 확인
print("\ngt_planning_command samples:")
for i in [0, 100, 500, 1000, 2000]:
    cmd = infos[i]['gt_planning_command']
    desc = infos[i].get('gt_planning_command_desc', 'N/A')
    print(f"  [{i}] cmd={cmd} (type={type(cmd).__name__}, shape={np.asarray(cmd).shape}), desc={desc!r}")

# 3. gt_attrs 샘플 확인 (Cursor가 나중에 활용할 field)
print("\ngt_attrs samples:")
for i in [0, 100]:
    attrs = infos[i].get('gt_attrs', None)
    names = infos[i].get('gt_names', None)
    if attrs is not None and len(attrs) > 0:
        print(f"  [{i}] first 3: names={names[:3]}, attrs={attrs[:3]}")

# 4. map_geoms 구조 확인
print("\nmap_geoms sample:")
mg = infos[0].get('map_geoms', None)
if mg is not None:
    if isinstance(mg, dict):
        print(f"  keys: {list(mg.keys())}")
        for k, v in mg.items():
            print(f"    {k}: type={type(v).__name__}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
    else:
        print(f"  type: {type(mg).__name__}, value preview: {str(mg)[:200]}")

mg = infos[0]['map_geoms']
for k, v in mg.items():
    print(f"\nkey={k}, num_items={len(v)}")
    if v:
        item = v[0]
        print(f"  type: {type(item).__name__}")
        if hasattr(item, 'shape'):
            print(f"  shape: {item.shape}")
            print(f"  dtype: {item.dtype}")
            print(f"  first 3 points: {item[:3]}")
        else:
            print(f"  content preview: {str(item)[:200]}")

# 여러 샘플에서 key 분포 확인
from collections import Counter
all_keys = Counter()
for info in infos[:100]:
    mg = info.get('map_geoms', {})
    if isinstance(mg, dict):
        all_keys.update(mg.keys())
print(f"\nKey distribution in first 100 samples: {all_keys}")