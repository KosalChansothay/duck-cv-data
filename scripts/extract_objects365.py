#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.objects365 import Objects365DuckExtractor, get_patch_url


def show_status(extractor: Objects365DuckExtractor, split: str):
    """Display real-time checkpoint and extraction status."""
    split_img_dir = extractor.images_dir / split
    checkpoint = extractor.get_checkpoint(split)
    
    ann_file = extractor.find_annotation_file(split)
    if not ann_file.exists():
        print(f"[Status] Annotation file not found for split '{split}'. Run extraction first.")
        return

    duck_images, duck_annotations = extractor.filter_ducks(ann_file)
    total_imgs = len(duck_images)
    total_boxes = sum(len(a) for a in duck_annotations.values())
    
    current_imgs = len(list(split_img_dir.glob("*.jpg"))) if split_img_dir.exists() else 0
    pct = (current_imgs / total_imgs * 100) if total_imgs else 0.0

    # Auto-sync ledger if empty
    if not checkpoint.data.get("completed_patches"):
        from collections import defaultdict
        patch_map = defaultdict(lambda: {"url": "", "files": set()})
        for img in duck_images.values():
            base_name = Path(img["file_name"]).name
            try:
                p_key, p_url = get_patch_url(img["file_name"], split=split)
                patch_map[p_key]["files"].add(base_name)
            except Exception:
                pass
        checkpoint.sync_from_disk(patch_map, split_img_dir)

    completed_patches = set(checkpoint.data.get("completed_patches", []))
    in_progress = checkpoint.data.get("in_progress_patch")
    last_updated = checkpoint.data.get("last_updated", "N/A")

    print("\n" + "=" * 65)
    print(f"  DUCK EXTRACTION STATUS: Objects365 ({split.upper()} SPLIT)")
    print("=" * 65)
    print(f"  • Total Duck Images Target:   {total_imgs:,}")
    print(f"  • Total Duck Bounding Boxes:  {total_boxes:,}")
    print(f"  • Images Currently On Disk:   {current_imgs:,} ({pct:.1f}%)")
    print(f"  • Completed Patches:          {len(completed_patches)} patches")
    print(f"  • In-Progress Patch:          {in_progress or 'None (idle)'}")
    print(f"  • Checkpoint Last Updated:    {last_updated}")
    print("=" * 65)
    print(f"\n[Tips for Background Execution]")
    print(f"  • WSL tmux:   wsl tmux new -s duck_extract 'python3 scripts/extract_objects365.py --split {split}'")
    print(f"  • Detach:     Press Ctrl+B then D")
    print(f"  • Reattach:   wsl tmux attach -t duck_extract")
    print(f"  • Windows:    Start-Process -NoNewWindow python -ArgumentList 'scripts/extract_objects365.py --split {split}'\n")


def main():
    parser = argparse.ArgumentParser(description="Extract Duck data from Objects365 with guarded checkpointing")
    parser.add_argument("--split", choices=["val", "train"], default="val", help="Dataset split to extract (default: val)")
    parser.add_argument("--dir", default=".", help="Root directory for dataset (default: current directory)")
    parser.add_argument("--status", action="store_true", help="Display current checkpoint and progress status without extracting")
    parser.add_argument("--max-patches", type=int, default=None, help="Maximum number of patches to process in this run")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of duck images to extract (optional cap)")
    parser.add_argument("--skip-images", action="store_true", help="Skip image extraction and only generate labels/summary")
    args = parser.parse_args()

    extractor = Objects365DuckExtractor(root_dir=args.dir)

    if args.status:
        show_status(extractor, split=args.split)
        return

    extractor.run_pipeline(
        split=args.split,
        max_images=args.max_images,
        max_patches=args.max_patches,
        skip_images=args.skip_images
    )


if __name__ == "__main__":
    main()
