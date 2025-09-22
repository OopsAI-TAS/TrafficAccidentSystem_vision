import os
import random

def keep_fraction(folder, ratio=0.25, exts=".mp4"):
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    n_total = len(files)
    n_keep = max(1, int(n_total * ratio))

    # 남길 파일만 랜덤 선택
    keep_files = set(random.sample(files, n_keep))

    removed = 0
    for f in files:
        if f not in keep_files:
            os.remove(os.path.join(folder, f))
            removed += 1

    print(f"{folder}: {n_total}개 중 {n_keep}개 유지, {removed}개 삭제")

# 여러 폴더 처리
root = "/Users/yoonsjin/Downloads/095.교통사고 영상 데이터/01.데이터/1.Training/원천데이터_231108_add"   # 데이터셋 루트 폴더
for sub in os.listdir(root):
    folder = os.path.join(root, sub)
    if os.path.isdir(folder):
        keep_fraction(folder, ratio=0.25)
# keep_fraction(folder=root, ratio=0.25)
