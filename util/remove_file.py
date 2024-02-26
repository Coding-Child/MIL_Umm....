import os
import shutil

def clear_directory(dir_path):
    # 디렉토리 내의 모든 파일과 서브디렉토리에 대해 반복
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            # 파일이면 삭제, 디렉토리이면 shutil.rmtree()를 사용하여 디렉토리와 그 내용 모두 삭제
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')