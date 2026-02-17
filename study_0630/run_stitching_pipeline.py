"""
전체 스티칭 파이프라인 실행 스크립트
1. superglue_matching.py 실행하여 호모그래피 행렬 계산
2. 배치 스티칭 실행
"""

import subprocess
import os
import sys

def run_command(command, description):
    """
    명령어 실행
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"실행 명령: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("성공!")
        if result.stdout:
            print("출력:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"실행 실패: {e}")
        if e.stdout:
            print("표준 출력:")
            print(e.stdout)
        if e.stderr:
            print("오류 출력:")
            print(e.stderr)
        return False

def main():
    """
    메인 실행 함수
    """
    print("SuperGlue 스티칭 파이프라인 실행")
    print("=" * 60)
    
    # 1단계: 호모그래피 행렬 계산
    print("\n1단계: 호모그래피 행렬 계산")
    success = run_command(
        "python superglue_matching.py",
        "SuperGlue 매칭 및 호모그래피 행렬 계산"
    )
    
    if not success:
        print("호모그래피 행렬 계산에 실패했습니다.")
        return
    
    # 호모그래피 행렬 파일 확인
    if not os.path.exists("homography_matrix.npy"):
        print("호모그래피 행렬 파일이 생성되지 않았습니다.")
        return
    
    print("호모그래피 행렬 파일이 생성되었습니다.")
    
    # 2단계: 배치 스티칭 실행
    print("\n2단계: 배치 스티칭 실행")
    success = run_command(
        "python batch_stitching_with_homography.py",
        "호모그래피 행렬을 사용한 배치 스티칭"
    )
    
    if success:
        print("\n" + "="*60)
        print("스티칭 파이프라인 완료!")
        print("="*60)
        print("생성된 파일들:")
        print("- homography_matrix.npy (호모그래피 행렬)")
        print("- output_images/stitched_batch/ (스티칭된 이미지들)")
        print("- output_images/stitched_video.mp4 (스티칭된 비디오)")
        print("- superglue_matching_result.png (매칭 결과)")
        print("- image_alignment_result.png (정렬 결과)")
    else:
        print("배치 스티칭에 실패했습니다.")

if __name__ == "__main__":
    main() 