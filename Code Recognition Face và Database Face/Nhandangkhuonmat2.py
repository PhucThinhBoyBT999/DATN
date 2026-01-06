import cv2
import numpy as np
import os
import time
import requests
import threading
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image


# CẤU HÌNH
BLYNK_TOKEN = "dRetcrvdh9fU4oY6Fd88XwqpBCCXNJ_5"
BLYNK_URL = f"https://blynk.cloud/external/api/update?token={BLYNK_TOKEN}"

ESP32CAM_IN = "http://192.168.137.68:81/stream"
ESP32CAM_OUT = "http://192.168.137.86:81/stream"

VPIN_FACE_ID = "V14"
VPIN_FACE_NAME = "V15"

# Google Sheets
GOOGLE_SHEETS_URL = "https://script.google.com/macros/s/AKfycbxX3sWzaTqYUfEfOXxgaTFvpt4El9pOfIRl8uy006DgPbpS3osfx6V14zHcMRJ03ull/exec?action=getUsers"

# FaceNet config
MEDIAPIPE_CONFIDENCE = 0.7
FACENET_THRESHOLD = 0.6
COOLDOWN_SECONDS = 20
CAPTURE_DELAY = 0.3

# KHỞI TẠO

# MediaPipe
mp_face_detection = mp.solutions.face_detection  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore

# FaceNet
print("🔄 Đang load FaceNet model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

facenet = InceptionResnetV1(pretrained='vggface2').eval()
print("✅ FaceNet model loaded!")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    clear_screen()
def print_menu():
    print_header()
    print("   Database: Google Sheets")
    print("\n  1.  Thu thập ảnh khuôn mặt")
    print("  2.  Train Model (Auto-detect)")  # ← SỬA: Gộp 2 thành 1
    print("  3.  Bật CẢ 2 CAMERA (IN + OUT)")
    print("  4.  Quản lý người dùng")
    print("  0.  Thoát")
    print("\n" + "═" * 50)


def send_face_to_blynk(face_id, name, is_check_out=False):
    """Gửi Face ID và tên lên Blynk"""
    try:
        name = str(name)
        face_id = int(face_id)

        if is_check_out:
            face_id = -abs(face_id)

        url_id = f"{BLYNK_URL}&{VPIN_FACE_ID}={face_id}"
        requests.get(url_id, timeout=1)

        url_name = f"{BLYNK_URL}&{VPIN_FACE_NAME}={name}"
        requests.get(url_name, timeout=1)

        action = "OUT" if is_check_out else "IN"
        print(f"✅ {action}: {name} (ID {face_id})")
        return True
    except:
        return False


def load_users_from_google_sheets():
    """Load users từ Google Sheets"""
    try:
        print(" Load Google Sheets...")

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
        }

        response = requests.get(GOOGLE_SHEETS_URL, timeout=10, headers=headers, allow_redirects=True)

        if response.status_code != 200:
            return {}

        data = response.json()
        users = {}
        users_list = []

        if data.get('status') == 'success':
            users_list = data.get('users', [])
        elif data.get('success') == True:
            data_obj = data.get('data', {})
            users_list = data_obj.get('users', [])

        for user in users_list:
            user_id = int(user.get('id', 0))
            if user_id > 0:
                users[user_id] = {
                    'name': user.get('name', 'Unknown'),
                    'rfid': user.get('rfid', '')
                }

        print(f"✅ {len(users)} người")
        return users

    except Exception as e:
        print(f" Lỗi: {e}")
        return {}


def add_user_to_file(user_id, name, rfid):
    """Backup local"""
    if not os.path.exists("users.txt"):
        with open("users.txt", "w", encoding="utf-8") as f:
            pass
    with open("users.txt", "a", encoding="utf-8") as f:
        f.write(f"{user_id},{name},{rfid}\n")


# MEDIAPIPE DETECTION

def detect_faces(frame, detector):
    """Detect faces - MediaPipe"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    faces = []
    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            width = min(int(bbox.width * w), w - x)
            height = min(int(bbox.height * h), h - y)

            if width > 50 and height > 50:
                faces.append((x, y, width, height))

    return faces


# FACENET FUNCTIONS
def get_embedding(face_bgr):
    """
    Tính FaceNet embedding cho 1 face (BGR từ OpenCV)
    Trả về: numpy array 128 chiều
    """
    try:
        # Chuyển BGR → RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Resize về 160x160 (yêu cầu của FaceNet)
        face_resized = cv2.resize(face_rgb, (160, 160))

        # Chuyển sang PIL Image
        face_pil = Image.fromarray(face_resized)

        # Chuẩn hóa ảnh: [0, 255] → [-1, 1]
        face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.unsqueeze(0).to(device)

        # Tính embedding
        with torch.no_grad():
            embedding = facenet(face_tensor).cpu().numpy()[0]

        # Chuẩn hóa L2
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        print(f" Embedding error: {e}")
        return None


def compare_embeddings(embedding1, embedding2):
    """So sánh 2 embeddings - trả về distance"""
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance

# YÊU CẦU 1: THU THẬP ẢNH

def collect_faces():
    print_header()
    print("     📸 THU THẬP ẢNH                          ")

    print("📡 Đang load danh sách từ Google Sheets...")
    users = load_users_from_google_sheets()

    if not users:
        print(" Không load được Google Sheets!")
        print("Nhập thủ công:")
        name = input("Tên: ").strip()
        rfid = input("RFID: ").strip()
        face_id = input("ID: ").strip()
    else:
        print(f"✅ Tìm thấy {len(users)} người\n")
        print("📋 DANH SÁCH:\n")
        print(f"{'ID':<5} {'Tên':<25} {'RFID':<15}")
        print("-" * 50)
        for uid, info in sorted(users.items()):
            print(f"{uid:<5} {info['name']:<25} {info['rfid']:<15}")

        print("\n💡 Nhập TÊN (phải giống Google Sheets):")
        name = input("Tên: ").strip()

        found_user = None
        for uid, info in users.items():
            if info['name'].lower() == name.lower():
                found_user = (uid, info)
                break

        if found_user:
            face_id, user_info = found_user
            rfid = user_info['rfid']
            print(f"✅ Tìm thấy: ID={face_id}, RFID={rfid}")
        else:
            print(f"⚠️  Không tìm thấy '{name}' trong Google Sheets!")
            print("💡 Nhập thông tin mới:")
            rfid = input("RFID: ").strip()
            face_id = input("ID: ").strip()

    try:
        face_id = int(face_id)
    except:
        print("❌ ID phải số!")
        input()
        return

    add_user_to_file(face_id, name, rfid)

    # ★★★ YÊU CẦU 2: CHỈ DÙNG CAMERA IN ★★★
    cam_url = ESP32CAM_IN
    cam_type = "IN"
    print(f"\n📹 Sử dụng Camera {cam_type}")

    folder_name = f"Mauanh/{name.replace(' ', '_')}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print(f" Thư mục: {folder_name}")

    cap = cv2.VideoCapture(cam_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f" Camera {cam_type} lỗi!")
        input()
        return

    detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=MEDIAPIPE_CONFIDENCE
    )  # type: ignore

    #  CHỜ PHÁT HIỆN KHUÔN MẶT
    print(f"\n Đang chờ phát hiện khuôn mặt...")
    print(f" Hãy đưa mặt vào trước camera!\n")

    face_detected = False

    # Vòng lặp chờ phát hiện khuôn mặt
    while not face_detected:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        faces = detect_faces(frame, detector)

        # Vẽ UI
        cv2.putText(frame, f"{cam_type}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(faces) == 0:
            status_text = "WAITING FOR FACE..."
            status_color = (0, 0, 255)  # Đỏ
            cv2.putText(frame, status_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        else:
            # ★ PHÁT HIỆN KHUÔN MẶT → THOÁT VÒNG LẶP ★
            status_text = "FACE DETECTED! Starting..."
            status_color = (0, 255, 0)  # Xanh
            cv2.putText(frame, status_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Vẽ khung quanh khuôn mặt
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            cv2.imshow(f'Camera {cam_type}', frame)
            cv2.waitKey(1000)  # Hiển thị 1 giây

            face_detected = True
            print("✅ Phát hiện khuôn mặt! Bắt đầu thu thập...\n")

        cv2.imshow(f'Camera {cam_type}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Đã hủy")
            detector.close()
            cap.release()
            cv2.destroyAllWindows()
            input()
            return

    # BẮT ĐẦU THU THẬP
    count, max_images = 0, 100
    last_capture_time = 0

    print(f"📸 Thu {max_images} ảnh - Q để thoát")
    print(f"💡 Giữ mặt trong khung hình, đợi chữ 'READY' màu xanh!\n")

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        current_time = time.time()
        faces = detect_faces(frame, detector)

        time_since_last = current_time - last_capture_time
        can_capture = (time_since_last >= CAPTURE_DELAY) and (count > 0 or time_since_last > 1.0)

        cv2.putText(frame, f"{cam_type}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {count}/{max_images}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(faces) == 0:
            status = "NO FACE"
            status_color = (0, 0, 255)
        elif can_capture:
            status = "READY"
            status_color = (0, 255, 0)
        else:
            cooldown = CAPTURE_DELAY - time_since_last
            status = f"Wait {cooldown:.1f}s"
            status_color = (0, 165, 255)

        cv2.putText(frame, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if len(faces) > 0 and can_capture:
            for (x, y, w, h) in faces:
                count += 1

                face_rgb = frame[y:y + h, x:x + w]
                filename = f"{folder_name}/User.{face_id}.{count}.jpg"
                cv2.imwrite(filename, face_rgb)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"OK {count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                print(f"\r✅ {count}/{max_images}", end="")

                last_capture_time = current_time

                if count >= max_images:
                    break
                break
        else:
            for (x, y, w, h) in faces:
                color = (0, 165, 255) if not can_capture else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow(f'Camera {cam_type}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\n✅ {count} ảnh → {folder_name}")
    input()

def train_model_auto(reset=False):

    print_header()
    print("╔" + "═" * 48 + "╗")
    print("║" + " " * 15 + "TRAIN MODEL" + " " * 22 + "║")
    print("╚" + "═" * 48 + "╝")
    print()



    db_path = "trainer/facenet_database.npy"

    if reset:
        # Người dùng chọn "Reset và train lại"
        if os.path.exists(db_path):
            os.remove(db_path)
            print("🗑️  Đã xóa database cũ")
        print("🔵 Chế độ: Train toàn bộ (reset)")
        print("─" * 50)
        mode = 1

    elif os.path.exists(db_path):

        old_db = np.load(db_path, allow_pickle=True).item()
        print("🟢 Chế độ: Train người mới (incremental)")
        print(f"   Phát hiện database cũ: {len(old_db)} users")
        print("─" * 50)
        mode = 2

    else:
        # Chưa có database → Train toàn bộ lần đầu
        print("🔵 Chế độ: Train toàn bộ (lần đầu)")
        print("─" * 50)
        mode = 1

    print()

    # GỌI HÀM TRAIN CHUNG
    start_time = time.time()

    result = train_model_unified(mode)

    if result:
        elapsed = time.time() - start_time
        print()
        print(f"⏱️  Thời gian: {elapsed:.2f} giây")
        print()

    input("\nEnter...")


def train_model_unified(mode):
    # KIỂM TRA ĐIỀU KIỆN ĐẦU VÀO
    # Kiểm tra thư mục Mauanh
    if not os.path.exists("Mauanh"):
        print(" Chưa có thư mục Mauanh!")
        print(" Hãy chọn [1] để thu thập ảnh trước")
        return False

    # Khởi tạo biến
    old_database = {}
    target_user_ids = set()

    # Xử lý theo chế độ
    if mode == 2:  # Người mới
        # Kiểm tra database cũ
        if not os.path.exists("trainer/facenet_database.npy"):
            print("❌ Chưa có database cũ!")
            print(" Hệ thống sẽ chuyển sang chế độ Train toàn bộ")
            mode = 1
        else:
            # Load database cũ
            print(" Load database cũ...")
            old_database = np.load("trainer/facenet_database.npy",
                                   allow_pickle=True).item()
            print(f"✅ Database cũ: {len(old_database)} users\n")

    # XÁC ĐỊNH DANH SÁCH USER CẦN TRAIN

    print(" Quét thư mục Mauanh/...")

    # Tìm tất cả user IDs trong Mauanh
    all_user_ids = set()
    for folder_name in os.listdir("Mauanh"):
        folder_path = os.path.join("Mauanh", folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("User.") and filename.endswith(".jpg"):
                    user_id = int(filename.split(".")[1])
                    all_user_ids.add(user_id)
        elif folder_name.startswith("User.") and folder_name.endswith(".jpg"):
            user_id = int(folder_name.split(".")[1])
            all_user_ids.add(user_id)

    if not all_user_ids:
        print("❌ Không tìm thấy ảnh nào trong Mauanh!")
        return False

    print(f"✅ Tìm thấy {len(all_user_ids)} users trong Mauanh\n")

    # Xác định users cần train
    if mode == 1:  # Toàn bộ
        target_user_ids = all_user_ids
        print(f"🎯 Sẽ train: TẤT CẢ {len(target_user_ids)} users")

    elif mode == 2:  # Người mới
        target_user_ids = all_user_ids - set(old_database.keys())

        if not target_user_ids:
            print(" Không có user mới!")
            print(" Tất cả users đã được train")
            return False

        print(f" Sẽ train: {len(target_user_ids)} users mới")
        print(f"   Users mới: {sorted(target_user_ids)}")

    print()

    # ĐỌC ẢNH VÀ TÍNH EMBEDDING

    print(" Đang đọc ảnh và tính embedding...")
    print("  Mỗi ảnh ~40ms, vui lòng đợi...\n")

    embeddings_dict = {}
    processed = 0
    total_images = 0

    # Đếm tổng số ảnh
    for folder_name in os.listdir("Mauanh"):
        folder_path = os.path.join("Mauanh", folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("User.") and filename.endswith(".jpg"):
                    user_id = int(filename.split(".")[1])
                    if user_id in target_user_ids:
                        total_images += 1
        elif folder_name.startswith("User.") and folder_name.endswith(".jpg"):
            user_id = int(folder_name.split(".")[1])
            if user_id in target_user_ids:
                total_images += 1

    print(f"📊 Tổng số ảnh cần xử lý: {total_images}\n")

    # Xử lý ảnh
    for folder_name in os.listdir("Mauanh"):
        folder_path = os.path.join("Mauanh", folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("User.") and filename.endswith(".jpg"):
                    user_id = int(filename.split(".")[1])

                    # Kiểm tra user có trong danh sách cần train không
                    if user_id not in target_user_ids:
                        continue  # Skip

                    path = os.path.join(folder_path, filename)
                    img = cv2.imread(path)
                    embedding = get_embedding(img)

                    if embedding is not None:
                        if user_id not in embeddings_dict:
                            embeddings_dict[user_id] = []
                        embeddings_dict[user_id].append(embedding)

                    processed += 1
                    print(f"\r Processing: {processed}/{total_images} "
                          f"({processed * 100 // total_images}%)", end="")

        elif folder_name.startswith("User.") and folder_name.endswith(".jpg"):
            user_id = int(folder_name.split(".")[1])

            # Kiểm tra user có trong danh sách cần train không
            if user_id not in target_user_ids:
                continue  # Skip

            path = os.path.join("Mauanh", folder_name)
            img = cv2.imread(path)
            embedding = get_embedding(img)

            if embedding is not None:
                if user_id not in embeddings_dict:
                    embeddings_dict[user_id] = []
                embeddings_dict[user_id].append(embedding)

            processed += 1
            print(f"\r Processing: {processed}/{total_images} "
                  f"({processed * 100 // total_images}%)", end="")

    print("\n")

    # TÍNH EMBEDDING TRUNG BÌNH

    print(" Tính embedding trung bình cho mỗi user...")

    new_embeddings = {}
    for user_id, embeddings in embeddings_dict.items():
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        new_embeddings[user_id] = mean_embedding
        print(f"   User {user_id}: {len(embeddings)} ảnh → 1 embedding")

    print()
    # LƯU DATABASE
    if mode == 1:  # Toàn bộ → Ghi đè
        final_database = new_embeddings
        print(f" Lưu database (ghi đè): {len(final_database)} users")

    elif mode == 2:  # Người mới → Merge
        final_database = {**old_database, **new_embeddings}
        print(f" Merge database:")
        print(f"   {len(old_database)} users cũ + "
              f"{len(new_embeddings)} users mới = "
              f"{len(final_database)} total")

    # Tạo thư mục nếu chưa có
    if not os.path.exists("trainer"):
        os.makedirs("trainer")

    # Lưu file
    np.save("trainer/facenet_database.npy",
            np.array(final_database, dtype=object))

    print(f" Đã lưu: trainer/facenet_database.npy")
    print(f" Hoàn thành!")

    if mode == 2:
        print(f"\n💡 Tiết kiệm: Chỉ train {total_images} ảnh thay vì toàn bộ!")

    return True


def remove_user_from_database(user_id):
    """Xóa user khỏi FaceNet database"""
    db_path = "trainer/facenet_database.npy"

    if not os.path.exists(db_path):
        print("⚠️  Database chưa tồn tại")
        return False

    try:
        # Load database
        database = np.load(db_path, allow_pickle=True).item()

        # Kiểm tra user có trong database không
        if user_id not in database:
            print(f"⚠️  User {user_id} không có trong database")
            return False

        # Xóa user
        del database[user_id]

        # Lưu lại database
        np.save(db_path, np.array(database, dtype=object))

        print(f"✅ Đã xóa User {user_id} khỏi database")
        print(f"   Database còn: {len(database)} users")
        return True

    except Exception as e:
        print(f"❌ Lỗi khi xóa database: {e}")
        return False

def recognition_dual_camera():
    print_header()
    print("        NHẬN DIỆN (FaceNet)                  ")

    if not os.path.exists("trainer/facenet_database.npy"):
        print("❌ Chưa train! Chọn [2]")
        input()
        return

    print("📂 Load FaceNet database...")
    database = np.load("trainer/facenet_database.npy", allow_pickle=True).item()
    print(f"✅ Loaded {len(database)} users")

    users = load_users_from_google_sheets()

    if not users:
        users = {}
        for user_id in database.keys():
            users[user_id] = {'name': f'User {user_id}', 'rfid': ''}

    print(f"✅ {len(users)} người")
    print(f"🔍 MediaPipe + FaceNet")
    print(f"🎯 Threshold: {FACENET_THRESHOLD}")
    print("🟢 START - Q thoát\n")

    stop_event = threading.Event()

    def camera_thread(cam_url, cam_type, is_checkout):
        cap = cv2.VideoCapture(cam_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"❌ Camera {cam_type} lỗi!")
            return

        detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=MEDIAPIPE_CONFIDENCE
        )  # type: ignore

        # ═══════════════════════════════════════════════════════
        # BIẾN TRẠNG THÁI
        # ═══════════════════════════════════════════════════════
        last_success_id = -1  # ID người cuối xác thực THÀNH CÔNG
        last_success_time = 0  # Thời gian xác thực thành công cuối

        first_detection_time = 0  # Thời gian phát hiện khuôn mặt lần đầu
        current_face_id = -1  # ID khuôn mặt hiện tại
        stranger_warned = False  # Đã cảnh báo người lạ chưa

        count = 0  # Số lần xác thực thành công
        fps, fc, lt = 0, 0, time.time()

        color = (0, 255, 255) if is_checkout else (0, 255, 0)

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            fc += 1
            ct = time.time()
            if ct - lt >= 1.0:
                fps, fc, lt = fc, 0, ct

            faces = detect_faces(frame, detector)

            # ═══════════════════════════════════════════════════════
            # TRƯỜNG HỢP 1: KHÔNG CÓ KHUÔN MẶT
            # ═══════════════════════════════════════════════════════
            if len(faces) == 0:
                # Reset trạng thái khi không có khuôn mặt
                first_detection_time = 0
                current_face_id = -1
                stranger_warned = False

                status_text = "No face detected"
                status_color = (128, 128, 128)
                cv2.putText(frame, status_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # ═══════════════════════════════════════════════════════
            # TRƯỜNG HỢP 2: CÓ KHUÔN MẶT
            # ═══════════════════════════════════════════════════════
            else:
                for (x, y, w, h) in faces:

                    # ───────────────────────────────────────────────
                    # BƯỚC 1: KIỂM TRA COOLDOWN (CHỈ KHI THÀNH CÔNG)
                    # ───────────────────────────────────────────────
                    time_since_success = ct - last_success_time
                    in_cooldown = (last_success_id != -1 and
                                   time_since_success <= COOLDOWN_SECONDS)

                    if in_cooldown:
                        # Hiển thị người vừa xác thực thành công
                        name = (users[last_success_id]['name']
                                if last_success_id in users else "Unknown")
                        cooldown_remaining = COOLDOWN_SECONDS - time_since_success
                        col = (0, 255, 0)
                        conf_text = f"{cooldown_remaining:.0f}s"

                        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
                        cv2.putText(frame, f"{name} (Wait {conf_text})",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, col, 2)
                        continue  # Bỏ qua xử lý

                    # ───────────────────────────────────────────────
                    # BƯỚC 2: NHẬN DIỆN KHUÔN MẶT
                    # ───────────────────────────────────────────────
                    face_bgr = frame[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                    embedding = get_embedding(face_rgb)

                    if embedding is None:
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 165, 255), 2)
                        cv2.putText(frame, "Processing...", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 165, 255), 2)
                        continue

                    # So sánh với database
                    min_distance = float('inf')
                    predicted_id = -1

                    for user_id, db_embedding in database.items():
                        distance = compare_embeddings(embedding, db_embedding)
                        if distance < min_distance:
                            min_distance = distance
                            predicted_id = user_id

                    # ───────────────────────────────────────────────
                    # BƯỚC 3: XỬ LÝ KẾT QUẢ NHẬN DIỆN
                    # ───────────────────────────────────────────────

                    # 3A: NHẬN DIỆN THÀNH CÔNG
                    if min_distance < FACENET_THRESHOLD and predicted_id in users:
                        name = users[predicted_id]['name']
                        conf_text = f"{(1 - min_distance) * 100:.0f}%"

                        # Kiểm tra xem đây có phải người mới không
                        if current_face_id != predicted_id:
                            # Người mới → Reset timer, gửi Blynk
                            first_detection_time = ct
                            current_face_id = predicted_id
                            stranger_warned = False

                            # GỬI LÊN BLYNK (XÁC THỰC 2 YẾU TỐ)
                            send_face_to_blynk(predicted_id, name, is_checkout)

                            # CẬP NHẬT THÀNH CÔNG (BẮT ĐẦU COOLDOWN)
                            last_success_id = predicted_id
                            last_success_time = ct
                            count += 1

                            print(f"✅ {cam_type}: {name} (ID {predicted_id})")

                        col = (0, 255, 0)

                    # 3B: NHẬN DIỆN THẤT BẠI (NGƯỜI LẠ)
                    else:
                        name = "Unknown"
                        col = (0, 0, 255)
                        conf_text = (f"{(1 - min_distance) * 100:.0f}%"
                                     if min_distance < 1 else "0%")

                        # Kiểm tra xem đây có phải người lạ mới không
                        if current_face_id != -2:  # -2 = người lạ
                            # Người lạ mới → Bắt đầu đếm thời gian
                            first_detection_time = ct
                            current_face_id = -2
                            stranger_warned = False

                        # Kiểm tra đã quá 20s chưa
                        time_since_first = ct - first_detection_time
                        if time_since_first >= 20 and not stranger_warned:
                            # CẢNH BÁO NGƯỜI LẠ
                            print(f"⚠️  {cam_type}: NGƯỜI LẠ xuất hiện > 20s!")

                            # Gửi cảnh báo lên Blynk (ID âm để ESP32 biết là người lạ)
                            send_face_to_blynk( "NGUOI LA", is_checkout)

                            stranger_warned = True  # Chỉ cảnh báo 1 lần

                    # ───────────────────────────────────────────────
                    # BƯỚC 4: VẼ KHUNG VÀ TEXT
                    # ───────────────────────────────────────────────
                    cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
                    cv2.putText(frame, f"{name} ({conf_text})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, col, 2)

            # ═══════════════════════════════════════════════════════
            # HIỂN THỊ THÔNG TIN
            # ═══════════════════════════════════════════════════════
            cv2.putText(frame, f"{cam_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"FPS: {fps}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Count: {count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(f'Camera {cam_type}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()

        detector.close()
        cap.release()

    t1 = threading.Thread(target=camera_thread,
                          args=(ESP32CAM_IN, "IN", False))
    t2 = threading.Thread(target=camera_thread,
                          args=(ESP32CAM_OUT, "OUT", True))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    cv2.destroyAllWindows()
    print("\n🔴 Tắt")
    input()


def manage_users():
    print_header()
    print("     QUẢN LÝ NGƯỜI DÙNG                  ")

    users = load_users_from_google_sheets()
    if not users:
        print("❌ Không load được Google Sheets!")
        input("\nEnter để quay lại...")
        return

    # Vòng lặp menu quản lý
    while True:
        print_header()
        print("     QUẢN LÝ NGƯỜI DÙNG                  ")
        print()

        # Hiển thị danh sách người dùng
        print("📋 DANH SÁCH NGƯỜI DÙNG:\n")
        print(f"{'ID':<5} {'Tên':<25} {'RFID':<15} {'Ảnh':<10}")
        print("-" * 60)

        for uid, info in sorted(users.items()):
            folder_name = f"Mauanh/{info['name'].replace(' ', '_')}"
            image_count = 0

            # Đếm ảnh trong folder
            if os.path.exists(folder_name) and os.path.isdir(folder_name):
                image_count = len([fn for fn in os.listdir(folder_name)
                                   if fn.startswith(f"User.{uid}.")])

            # Đếm ảnh trong Mauanh (nếu có ảnh rời)
            if os.path.exists("Mauanh"):
                image_count += len([fn for fn in os.listdir("Mauanh")
                                    if fn.startswith(f"User.{uid}.") and
                                    os.path.isfile(os.path.join("Mauanh", fn))])

            print(f"{uid:<5} {info['name']:<25} {info['rfid']:<15} {image_count} ảnh")

        print("\n" + "═" * 60)
        print("\n  1. Xem ảnh")
        print("  2. Xóa người dùng")
        print("  0. Quay lại")
        print("\n" + "═" * 60)

        choice = input("\nChọn: ").strip()

        # ═══════════════════════════════════════════════════════
        # NHÁNH 1: XEM ẢNH
        # ═══════════════════════════════════════════════════════
        if choice == '1':
            try:
                uid = int(input("\nNhập ID để xem ảnh: ").strip())

                # Kiểm tra ID có tồn tại không
                if uid not in users:
                    print(f"\n❌ Thông báo: ID {uid} không tồn tại trong danh sách!")
                    input("\nEnter để tiếp tục...")
                    continue  # Quay lại hiển thị menu

                # Tìm ảnh
                images = []
                folder_name = f"Mauanh/{users[uid]['name'].replace(' ', '_')}"

                # Tìm trong folder
                if os.path.exists(folder_name) and os.path.isdir(folder_name):
                    images += [os.path.join(folder_name, fn)
                               for fn in os.listdir(folder_name)
                               if fn.startswith(f"User.{uid}.")]

                # Tìm ảnh rời trong Mauanh
                if os.path.exists("Mauanh"):
                    images += [os.path.join("Mauanh", fn)
                               for fn in os.listdir("Mauanh")
                               if fn.startswith(f"User.{uid}.") and
                               os.path.isfile(os.path.join("Mauanh", fn))]

                # Kiểm tra có ảnh không
                if not images:
                    print(f"\n⚠️  Thông báo: Chưa thu thập ảnh cho '{users[uid]['name']}'!")
                    print("💡 Hãy chọn [1] Thu thập ảnh khuôn mặt để thu thập ảnh")
                    input("\nEnter để tiếp tục...")
                    continue  # Quay lại hiển thị menu

                # Hiển thị ảnh
                print(f"\n✅ Tìm thấy {len(images)} ảnh của '{users[uid]['name']}'")
                print("💡 Nhấn SPACE để xem ảnh tiếp theo, Q để thoát\n")

                idx = 0
                while True:
                    img = cv2.imread(images[idx])
                    if img is not None:
                        img = cv2.resize(img, (400, 400))
                        cv2.putText(
                            img,
                            f"{users[uid]['name']} ({idx + 1}/{len(images)})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2
                        )
                        cv2.imshow('Xem anh', img)

                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        idx = (idx + 1) % len(images)

                cv2.destroyAllWindows()
                input("\nEnter để tiếp tục...")

            except ValueError:
                print("\n❌ ID phải là số!")
                input("\nEnter để tiếp tục...")
            except Exception as e:
                print(f"\n❌ Lỗi: {e}")
                input("\nEnter để tiếp tục...")

                # ═══════════════════════════════════════════════════════
                # NHÁNH 2: XÓA NGƯỜI DÙNG
                # ═══════════════════════════════════════════════════════
        elif choice == '2':
                try:
                    uid = int(input("\nNhập ID để xóa: ").strip())

                    # Kiểm tra ID có tồn tại không
                    if uid not in users:
                        print(f"\n❌ Thông báo: ID {uid} không tồn tại trong danh sách!")
                        input("\nEnter để tiếp tục...")
                        continue

                    # Hiển thị thông tin người dùng
                    print(f"\n{'=' * 60}")
                    print(f"📋 THÔNG TIN NGƯỜI DÙNG:")
                    print(f"   ID: {uid}")
                    print(f"   Tên: {users[uid]['name']}")
                    print(f"   RFID: {users[uid]['rfid']}")

                    # Đếm ảnh
                    folder_name = f"Mauanh/{users[uid]['name'].replace(' ', '_')}"
                    image_count = 0

                    if os.path.exists(folder_name) and os.path.isdir(folder_name):
                        image_count = len([fn for fn in os.listdir(folder_name)
                                           if fn.startswith(f"User.{uid}.")])

                    if os.path.exists("Mauanh"):
                        image_count += len([fn for fn in os.listdir("Mauanh")
                                            if fn.startswith(f"User.{uid}.") and
                                            os.path.isfile(os.path.join("Mauanh", fn))])

                    print(f"   Số ảnh: {image_count}")

                    # Kiểm tra có trong database không
                    db_exists = os.path.exists("trainer/facenet_database.npy")
                    in_database = False

                    if db_exists:
                        database = np.load("trainer/facenet_database.npy",
                                           allow_pickle=True).item()
                        in_database = uid in database
                        print(f"   Trong database: {'Có' if in_database else 'Không'}")

                    print(f"{'=' * 60}\n")

                    # Xác nhận xóa
                    print("⚠️  BẠN SẼ XÓA:")
                    print(f"   ✓ Tất cả {image_count} ảnh")
                    if in_database:
                        print(f"   ✓ Embedding trong database (hệ thống sẽ KHÔNG nhận diện được nữa)")
                    print(f"\n💡 Lưu ý: Dữ liệu trên Google Sheets KHÔNG bị xóa")
                    print(f"         (chỉ xóa ảnh và khả năng nhận diện)\n")

                    confirm = input("Nhập 'YES' (viết hoa) để xác nhận: ").strip()

                    if confirm != 'YES':
                        print("\n✅ Đã hủy xóa")
                        input("\nEnter để tiếp tục...")
                        continue

                    # ═══════════════════════════════════════════════════════
                    # THỰC HIỆN XÓA
                    # ═══════════════════════════════════════════════════════
                    print(f"\n🔄 Đang xóa...")

                    deleted_count = 0

                    # 1. Xóa folder ảnh
                    if os.path.exists(folder_name) and os.path.isdir(folder_name):
                        import shutil
                        file_count = len([fn for fn in os.listdir(folder_name)
                                          if fn.startswith(f"User.{uid}.")])
                        shutil.rmtree(folder_name)
                        deleted_count += file_count
                        print(f"   ✓ Xóa thư mục: {folder_name}")

                    # 2. Xóa ảnh rời
                    if os.path.exists("Mauanh"):
                        for fn in os.listdir("Mauanh"):
                            if (fn.startswith(f"User.{uid}.") and
                                    os.path.isfile(os.path.join("Mauanh", fn))):
                                os.remove(os.path.join("Mauanh", fn))
                                deleted_count += 1

                    print(f"   ✓ Xóa {deleted_count} ảnh")

                    # 3. Xóa khỏi database
                    if in_database:
                        success = remove_user_from_database(uid)
                        if success:
                            print(f"   ✓ Xóa embedding khỏi database")
                        else:
                            print(f"   ⚠️  Không xóa được database")

                    # KẾT QUẢ
                    print(f"\n{'=' * 60}")
                    print(f"✅ XÓA THÀNH CÔNG!")
                    print(f"   • Đã xóa {deleted_count} ảnh")
                    if in_database:
                        print(f"   • Hệ thống sẽ KHÔNG nhận diện '{users[uid]['name']}' nữa")
                    print(f"   • Dữ liệu Google Sheets vẫn còn (nếu cần xóa hẳn)")
                    print(f"{'=' * 60}")

                    input("\nEnter để tiếp tục...")

                except ValueError:
                    print("\n❌ ID phải là số!")
                    input("\nEnter để tiếp tục...")
                except Exception as e:
                    print(f"\n❌ Lỗi: {e}")
                    import traceback
                    traceback.print_exc()
                    input("\nEnter để tiếp tục...")

        # ═══════════════════════════════════════════════════════
        # NHÁNH 3: QUAY LẠI MENU CHÍNH
        # ═══════════════════════════════════════════════════════
        elif choice == '0':
            break  # Thoát vòng lặp, quay về menu chính

        else:
            print("\n❌ Lựa chọn không hợp lệ!")
            input("\nEnter để tiếp tục...")
# MAIN
def main():
    while True:
        print_menu()
        choice = input("\nChọn: ").strip()

        if choice == '1':
            collect_faces()

        elif choice == '2':
            train_model_auto(reset=False)  # Auto-detect

        elif choice == '3':
            recognition_dual_camera()

        elif choice == '4':
            manage_users()

        elif choice == '0':
            print("\n👋 Bye!")
            break

        else:
            print("❌ Lựa chọn không hợp lệ!")
            input()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Stop!")
    finally:
        cv2.destroyAllWindows()