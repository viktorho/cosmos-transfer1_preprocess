import cv2
import os

def check_mp4_valid(filepath: str) -> bool:
    """
    Kiểm tra file MP4 có hoạt động (có thể đọc được khung hình) hay không.

    Args:
        filepath (str): Đường dẫn đến file .mp4

    Returns:
        bool: True nếu file đọc được ít nhất 1 frame, False nếu hỏng hoặc không thể mở.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File không tồn tại: {filepath}")
        return False

    if not filepath.lower().endswith(".mp4"):
        return False

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở video: {filepath}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[ERROR] Không đọc được khung hình nào từ video: {filepath}")
        return False

    print(f"[OK] File '{os.path.basename(filepath)}' hoạt động tốt.")
    return True


def snapshot_first_frame(filepath: str, output_dir: str = "./snapshots") -> str | None:
    """
    Lưu khung hình đầu tiên của video ra file ảnh (snapshot).

    Args:
        filepath (str): Đường dẫn đến file .mp4
        output_dir (str): Thư mục lưu ảnh snapshot

    Returns:
        str | None: Đường dẫn đến file snapshot nếu thành công, None nếu lỗi.
    """
    if not check_mp4_valid(filepath):
        return None

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(filepath)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[ERROR] Không thể đọc khung hình đầu tiên từ {filepath}")
        return None

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    snapshot_path = os.path.join(output_dir, f"{base_name}_snapshot.jpg")

    success = cv2.imwrite(snapshot_path, frame)
    if success:
        print(f"[OK] Snapshot đã được lưu tại: {snapshot_path}")
        return snapshot_path
    else:
        print(f"[ERROR] Không thể lưu snapshot cho video: {filepath}")
        return None


def check_all_videos_in_folder(folder_path: str, save_snapshots: bool = False):
    """
    Duyệt toàn bộ folder, kiểm tra tất cả file mp4, và báo kết quả.

    Args:
        folder_path (str): Thư mục chứa video.
        save_snapshots (bool): Nếu True, tự động lưu khung hình đầu tiên cho video hợp lệ.
    """
    if not os.path.exists(folder_path):
        print(f"[ERROR] Thư mục không tồn tại: {folder_path}")
        return

    mp4_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".mp4")
    ]

    if not mp4_files:
        print(f"[INFO] Không có file .mp4 nào trong thư mục: {folder_path}")
        return

    print(f"[INFO] Tìm thấy {len(mp4_files)} file mp4, bắt đầu kiểm tra...\n")

    valid_videos = []
    invalid_videos = []

    for file in mp4_files:
        print(f"→ Đang kiểm tra: {file}")
        if check_mp4_valid(file):
            valid_videos.append(file)
            if save_snapshots:
                snapshot_first_frame(file)
        else:
            invalid_videos.append(file)
        print("-" * 60)

    print("\n📊 KẾT QUẢ TỔNG HỢP:")
    print(f"✅ Hợp lệ: {len(valid_videos)} file")
    print(f"❌ Lỗi: {len(invalid_videos)} file")

    if invalid_videos:
        print("\nDanh sách file lỗi:")
        for f in invalid_videos:
            print("  -", f)

    return valid_videos, invalid_videos