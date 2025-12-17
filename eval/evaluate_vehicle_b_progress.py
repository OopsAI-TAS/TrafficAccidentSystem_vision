"""
VehicleBProgress Rule-based 시스템 성능 평가 스크립트

사용법:
1. Ground truth 라벨 파일 준비 (JSON 또는 CSV 형식)
2. 예측 결과 파일 준비
3. 정확도, Precision, Recall, F1-score, Confusion Matrix 계산
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# VehicleBProgress 클래스 정의
VEHICLE_B_CLASSES = {
    "0": "stopping",
    "1": "starting", 
    "2": "left_turn",
    "3": "right_turn",
    "4": "straight",
}

CLASS_NAMES = ["stopping", "starting", "left_turn", "right_turn", "straight"]


def load_mapping(mapping_json: str):
    """
    mapping.json 로드하여 원본 코드 → 클래스 인덱스 변환 딕셔너리 생성
    
    Args:
        mapping_json: configs/mapping.json 파일 경로
    
    Returns:
        dict: {원본_코드: 클래스_인덱스}
        예: {1: 4, 2: 4, 3: 3, ...}  # 1번은 straight(4), 3번은 right_turn(3)
    """
    mapping_path = Path(mapping_json)
    
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.json을 찾을 수 없습니다: {mapping_path}")
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # mapping.json: 최상위에 vehicle_b_progress_info
    vb_mapping = mapping_data.get('vehicle_b_progress_info', {})
    
    # 클래스명 → 인덱스
    class_to_idx = {
        "stopping": 0,
        "starting": 1,
        "left_turn": 2,
        "right_turn": 3,
        "straight": 4
    }
    
    # 원본 코드 → 클래스 인덱스 매핑 생성
    code_to_class_idx = {}
    
    for class_name, codes in vb_mapping.items():
        class_idx = class_to_idx.get(class_name)
        if class_idx is not None:
            for code in codes:
                code_to_class_idx[code] = class_idx
    
    return code_to_class_idx


def load_ground_truth(gt_dir: str, mapping_json: str, video_list_txt: str = None):
    """
    Ground truth 라벨 로드 및 mapping.json 적용
    
    각 비디오명.json 파일에서 vehicle_b_progress_info 읽고,
    mapping.json을 사용하여 클래스 인덱스로 변환
    
    Args:
        gt_dir: Ground truth JSON 파일들이 있는 디렉토리
        mapping_json: configs/mapping.json 파일 경로
        video_list_txt: 선택적, 비디오 리스트 txt 파일
    
    Returns:
        dict: {video_name: {"vehicle_b_progress": str}}
    """
    gt_dir = Path(gt_dir)
    
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth 디렉토리를 찾을 수 없습니다: {gt_dir}")
    
    # 매핑 로드
    code_to_class_idx = load_mapping(mapping_json)
    print(f"✓ Mapping 로드 완료: {len(code_to_class_idx)}개 코드 매핑")
    
    # 비디오 리스트 로드 (있으면)
    video_list = None
    if video_list_txt:
        video_list_path = Path(video_list_txt)
        if video_list_path.exists():
            with open(video_list_path, 'r', encoding='utf-8') as f:
                video_list = set(line.strip() for line in f if line.strip())
    
    ground_truth = {}
    unmapped_codes = set()
    
    # 디렉토리 내 모든 JSON 파일 읽기
    for json_file in gt_dir.glob("*.json"):
        # JSON: video001.json → 비디오명: video001.mp4
        video_name = f"{json_file.stem}.mp4"
        
        # video_list가 있으면 필터링
        if video_list and video_name not in video_list:
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 레이블 JSON: video 안에 vehicle_b_progress_info
            original_code = data.get('video', {}).get('vehicle_b_progress_info', -1)
            
            # mapping.json을 사용하여 클래스 인덱스로 변환
            if original_code in code_to_class_idx:
                class_idx = code_to_class_idx[original_code]
            else:
                # 매핑에 없는 코드는 -1로 처리
                class_idx = -1
                unmapped_codes.add(original_code)
            
            ground_truth[video_name] = {
                'vehicle_b_progress': str(class_idx)
            }
        except Exception as e:
            print(f"⚠️  파일 로드 실패: {json_file} - {e}")
    
    # 매핑되지 않은 코드가 있으면 경고
    if unmapped_codes:
        print(f"⚠️  매핑되지 않은 코드 발견: {sorted(unmapped_codes)}")
    
    return ground_truth


def load_predictions(pred_json: str):
    """
    예측 결과 로드 (하나의 JSON 파일에서)
    
    Args:
        pred_json: 예측 결과 JSON 파일 경로
    
    Returns:
        dict: {video_name: {"vehicle_b_progress": str}}
    """
    pred_json = Path(pred_json)
    
    if not pred_json.exists():
        raise FileNotFoundError(f"예측 결과 파일을 찾을 수 없습니다: {pred_json}")
    
    with open(pred_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = {}
    
    # 각 비디오에 대해 vehicle_b_progress_info 추출
    for video_name, pred_data in data.items():
        vb_progress = pred_data.get('vehicle_b_progress_info', -1)
        predictions[video_name] = {
            'vehicle_b_progress': str(vb_progress)
        }
    
    return predictions


def evaluate_performance(ground_truth, predictions, output_dir="evaluation_results"):
    """
    성능 평가 수행
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 매칭되는 비디오만 추출
    common_videos = set(ground_truth.keys()) & set(predictions.keys())
    
    if len(common_videos) == 0:
        print("❌ Ground truth와 예측 결과에 공통된 비디오가 없습니다!")
        return
    
    print(f"✓ 평가 대상 비디오 개수: {len(common_videos)}")
    print(f"  - Ground truth: {len(ground_truth)}")
    print(f"  - Predictions: {len(predictions)}")
    
    # 라벨과 예측값 추출
    y_true = []
    y_pred = []
    video_names = []
    
    for video_name in sorted(common_videos):
        gt_label = ground_truth[video_name]['vehicle_b_progress']
        pred_label = predictions[video_name]['vehicle_b_progress']
        
        # 유효한 클래스만 포함 (0~4)
        if gt_label in VEHICLE_B_CLASSES and pred_label in VEHICLE_B_CLASSES:
            y_true.append(int(gt_label))
            y_pred.append(int(pred_label))
            video_names.append(video_name)
    
    if len(y_true) == 0:
        print("❌ 유효한 평가 데이터가 없습니다!")
        return
    
    print(f"✓ 유효한 샘플 개수: {len(y_true)}")
    
    # === 1. 전체 성능 지표 계산 ===
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print("\n" + "="*60)
    print("전체 성능 지표")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall:    {recall:.4f} (weighted)")
    print(f"F1-Score:  {f1:.4f} (weighted)")
    
    # === 2. 클래스별 성능 지표 ===
    print("\n" + "="*60)
    print("클래스별 성능 지표")
    print("="*60)
    report = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    )
    print(report)
    
    # 리포트를 파일로 저장
    with open(output_dir / "classification_report.txt", 'w', encoding='utf-8') as f:
        f.write("전체 성능 지표\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f} (weighted)\n")
        f.write(f"Recall:    {recall:.4f} (weighted)\n")
        f.write(f"F1-Score:  {f1:.4f} (weighted)\n\n")
        f.write("클래스별 성능 지표\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    # === 3. Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix - VehicleBProgress', fontsize=16, pad=20)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    print(f"\n✓ Confusion matrix 저장: {output_dir / 'confusion_matrix.png'}")
    
    # === 4. 오분류 분석 ===
    print("\n" + "="*60)
    print("오분류 분석")
    print("="*60)
    
    misclassified = []
    for i, (video, gt, pred) in enumerate(zip(video_names, y_true, y_pred)):
        if gt != pred:
            misclassified.append({
                'video_name': video,
                'ground_truth': VEHICLE_B_CLASSES[str(gt)],
                'predicted': VEHICLE_B_CLASSES[str(pred)],
                'gt_label': gt,
                'pred_label': pred
            })
    
    print(f"총 오분류 개수: {len(misclassified)} / {len(y_true)} ({len(misclassified)/len(y_true)*100:.2f}%)")
    
    if len(misclassified) > 0:
        # 오분류 패턴 분석
        error_patterns = Counter()
        for item in misclassified:
            pattern = f"{item['ground_truth']} → {item['predicted']}"
            error_patterns[pattern] += 1
        
        print("\n주요 오분류 패턴:")
        for pattern, count in error_patterns.most_common(10):
            print(f"  {pattern}: {count}회")
        
        # 오분류 목록 저장
        df_errors = pd.DataFrame(misclassified)
        df_errors.to_csv(output_dir / "misclassified_samples.csv", index=False, encoding='utf-8')
        print(f"\n✓ 오분류 목록 저장: {output_dir / 'misclassified_samples.csv'}")
    
    # === 5. 클래스 분포 비교 ===
    gt_dist = Counter(y_true)
    pred_dist = Counter(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ground truth 분포
    axes[0].bar([VEHICLE_B_CLASSES[str(i)] for i in range(5)], 
                [gt_dist[i] for i in range(5)], 
                color='steelblue', alpha=0.7)
    axes[0].set_title('Ground Truth Distribution', fontsize=14)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Prediction 분포
    axes[1].bar([VEHICLE_B_CLASSES[str(i)] for i in range(5)], 
                [pred_dist[i] for i in range(5)], 
                color='coral', alpha=0.7)
    axes[1].set_title('Prediction Distribution', fontsize=14)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    print(f"✓ 클래스 분포 저장: {output_dir / 'class_distribution.png'}")
    
    # === 6. 종합 결과 JSON 저장 ===
    results = {
        'total_samples': len(y_true),
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'misclassified_count': len(misclassified),
        'class_distribution': {
            'ground_truth': {VEHICLE_B_CLASSES[str(k)]: int(v) for k, v in gt_dist.items()},
            'predictions': {VEHICLE_B_CLASSES[str(k)]: int(v) for k, v in pred_dist.items()}
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 평가 결과 요약 저장: {output_dir / 'evaluation_summary.json'}")
    print("\n" + "="*60)
    print("평가 완료!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="VehicleBProgress Rule-based 시스템 성능 평가")
    parser.add_argument("--gt_dir", type=str, required=True, 
                        help="Ground truth JSON 파일들이 있는 디렉토리")
    parser.add_argument("--pred_json", type=str, required=True,
                        help="예측 결과 JSON 파일 경로")
    parser.add_argument("--mapping_json", type=str, default="configs/mapping.json",
                        help="클래스 매핑 JSON 파일 경로 (기본: configs/mapping.json)")
    parser.add_argument("--video_list", type=str, default=None,
                        help="선택적, 평가할 비디오 리스트 txt 파일")
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="평가 결과를 저장할 디렉토리")
    
    args = parser.parse_args()
    
    print("="*60)
    print("VehicleBProgress 성능 평가")
    print("="*60)
    print(f"Ground Truth Dir: {args.gt_dir}")
    print(f"Predictions JSON: {args.pred_json}")
    print(f"Mapping JSON:     {args.mapping_json}")
    if args.video_list:
        print(f"Video List:       {args.video_list}")
    print(f"Output Dir:       {args.output}")
    print("="*60 + "\n")
    
    # Ground truth 로드 (mapping 적용)
    ground_truth = load_ground_truth(args.gt_dir, args.mapping_json, args.video_list)
    print(f"✓ Ground truth 로드 완료: {len(ground_truth)}개 비디오\n")
    
    # 예측 결과 로드
    predictions = load_predictions(args.pred_json)
    print(f"✓ 예측 결과 로드 완료: {len(predictions)}개 비디오\n")
    
    # 성능 평가
    evaluate_performance(ground_truth, predictions, args.output)


if __name__ == "__main__":
    main()
