#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path

# 데이터로 인정할 확장자 (필요하면 추가)
VIDEO_EXTS = '.mp4'
DATA_EXTS  = VIDEO_EXTS

def index_data_basenames(data_dir: Path):
    names = set()
    for p in data_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in DATA_EXTS:
            names.add(p.stem)  # 확장자 제외한 파일명
    return names

def prune_label_files(label_dir: Path, data_names: set, dry_run=True, move_to: Path|None=None):
    total = removed = 0
    for lp in label_dir.rglob('*.json'):
        total += 1
        if lp.stem not in data_names:
            removed += 1
            if dry_run:
                print(f'[DRY] would remove: {lp}')
            else:
                if move_to:
                    dst = move_to / lp.relative_to(label_dir)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(lp), str(dst))
                    print(f'moved: {lp} -> {dst}')
                else:
                    lp.unlink()
                    print(f'deleted: {lp}')
    print(f'\nChecked {total} label files; {"would remove" if dry_run else "removed"} {removed}.')

def main():
    ap = argparse.ArgumentParser(description='Delete/move label JSONs without corresponding data files.')
    ap.add_argument('--data_dir',  required=True, help='이미지/영상이 있는 루트 디렉토리')
    ap.add_argument('--label_dir', required=True, help='라벨(JSON) 루트 디렉토리')
    ap.add_argument('--apply', action='store_true', help='실제로 삭제/이동 수행 (기본: DRY RUN)')
    ap.add_argument('--backup_dir', type=str, help='설정 시 삭제 대신 여기에 이동')
    args = ap.parse_args()

    data_dir  = Path(args.data_dir).resolve()
    label_dir = Path(args.label_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None

    data_names = index_data_basenames(data_dir)
    print(f'Indexed {len(data_names)} data basenames from: {data_dir}')
    prune_label_files(label_dir, data_names, dry_run=not args.apply, move_to=backup_dir)

if __name__ == '__main__':
    main()
