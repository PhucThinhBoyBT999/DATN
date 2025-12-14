import cv2
import numpy as np
import os
import time
import requests
import threading
import mediapipe as mp
from keras_facenet import FaceNet

# ═══════════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# KHỞI TẠO
# ═══════════════════════════════════════════════════════════════════════════

# MediaPipe
mp_face_detection = mp.solutions.face_detection  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore

# FaceNet
print("🔄 Đang load FaceNet model...")
facenet = FaceNet()
print("✅ FaceNet model loaded!")


# ═══════════════════════════════════════════════════════════════════════════
# TIỆN ÍCH
# ═══════════════════════════════════════════════════════════════════════════

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    clear_screen()
    print("╔═══════════════════════════════════════════════╗")
    print("║   NHẬN DIỆN - MEDIAPIPE + FACENET            ║")
    print("╚═══════════════════════════════════════════════╝")


def print_menu():
    print_header()
    print("  🎯 Database: Google Sheets")
    print("\n  1.  Thu thập ảnh khuôn mặt")
    print("  2.  Huấn luyện FaceNet (toàn bộ)")
    print("  3.  Huấn luyện FaceNet (chỉ người mới) ⚡")
    print("  4.  Bật CẢ 2 CAMERA (IN + OUT)")
    print("  5.  Quản lý người dùng")
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
        print("📡 Load Google Sheets...")

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
        print(f"❌ Lỗi: {e}")
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
def get_embedding(face_rgb):
    """Tính FaceNet embedding cho 1 face (RGB)"""
    try:
        face_resized = cv2.resize(face_rgb, (160, 160))
        embedding = facenet.embeddings([face_resized])[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None


def compare_embeddings(embedding1, embedding2):
    """So sánh 2 embeddings - trả về distance"""
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance
# THU THẬP
def collect_faces():
    print_header()
    print("     📸 THU THẬP ẢNH                          ")

    print("📡 Đang load danh sách từ Google Sheets...")
    users = load_users_from_google_sheets()

    if not users:
        print("❌ Không load được Google Sheets!")
        print("💡 Nhập thủ công:")
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

    print("\n1. Camera IN | 2. Camera OUT")
    choice = input("Chọn: ").strip()

    cam_url = ESP32CAM_IN if choice == "1" else ESP32CAM_OUT
    cam_type = "IN" if choice == "1" else "OUT"

    folder_name = f"Mauanh/{name.replace(' ', '_')}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print(f"📁 Thư mục: {folder_name}")

    cap = cv2.VideoCapture(cam_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Camera {cam_type} lỗi!")
        input()
        return

    detector = mp_face_detection.FaceDetection(model_selection=0,
                                               min_detection_confidence=MEDIAPIPE_CONFIDENCE)  # type: ignore

    count, max_images = 0, 100
    last_capture_time = 0

    print(f"\n📸 Thu {max_images} ảnh - Q thoát")
    print(f"💡 Đưa mặt vào camera, đợi chữ 'READY' màu xanh!\n")

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        current_time = time.time()
        faces = detect_faces(frame, detector)

        time_since_last = current_time - last_capture_time
        can_capture = (time_since_last >= CAPTURE_DELAY) and (count > 0 or time_since_last > 1.0)

        cv2.putText(frame, f"{cam_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {count}/{max_images}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

        cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if len(faces) > 0 and can_capture:
            for (x, y, w, h) in faces:
                count += 1

                face_rgb = frame[y:y + h, x:x + w]
                filename = f"{folder_name}/User.{face_id}.{count}.jpg"
                cv2.imwrite(filename, face_rgb)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"OK {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

# TRAINING FACENET (TOÀN BỘ)

def train_model():
    print_header()
    print("     TRAINING FACENET (TOÀN BỘ)           ")

    if not os.path.exists("Mauanh"):
        print("❌ Chưa có ảnh!")
        input()
        return

    print("📂 Đọc ảnh từ các thư mục...")
    print("⏱️  Mỗi ảnh ~40ms, vui lòng đợi...\n")

    embeddings_dict = {}
    total_images = 0
    start_time = time.time()

    # Đếm tổng số ảnh
    for folder_name in os.listdir("Mauanh"):
        folder_path = os.path.join("Mauanh", folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("User.") and filename.endswith(".jpg"):
                    total_images += 1
        elif folder_name.startswith("User.") and folder_name.endswith(".jpg"):
            total_images += 1

    if total_images == 0:
        print("❌ Không có ảnh!")
        input()
        return

    print(f"📊 Tìm thấy {total_images} ảnh\n")

    processed = 0

    # Đọc từ folders
    for folder_name in os.listdir("Mauanh"):
        folder_path = os.path.join("Mauanh", folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("User.") and filename.endswith(".jpg"):
                    path = os.path.join(folder_path, filename)
                    user_id = int(filename.split(".")[1])

                    img = cv2.imread(path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    embedding = get_embedding(img_rgb)

                    if embedding is not None:
                        if user_id not in embeddings_dict:
                            embeddings_dict[user_id] = []
                        embeddings_dict[user_id].append(embedding)

                    processed += 1
                    print(f"\r🔄 Processing: {processed}/{total_images} ({processed * 100 // total_images}%)", end="")

        elif folder_name.startswith("User.") and folder_name.endswith(".jpg"):
            path = os.path.join("Mauanh", folder_name)
            user_id = int(folder_name.split(".")[1])

            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(img_rgb)

            if embedding is not None:
                if user_id not in embeddings_dict:
                    embeddings_dict[user_id] = []
                embeddings_dict[user_id].append(embedding)

            processed += 1
            print(f"\r🔄 Processing: {processed}/{total_images} ({processed * 100 // total_images}%)", end="")

    print("\n\n📊 Tính embeddings trung bình cho mỗi user...")

    final_database = {}
    for user_id, embeddings in embeddings_dict.items():
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        final_database[user_id] = mean_embedding
        print(f"   User {user_id}: {len(embeddings)} ảnh → 1 embedding (512-dim)")

    if not os.path.exists("trainer"):
        os.makedirs("trainer")

    np.save("trainer/facenet_database.npy", np.array(final_database, dtype=object))

    elapsed = time.time() - start_time
    print(f"\n✅ Xong! File: trainer/facenet_database.npy")
    print(f"⏱️  Thời gian: {elapsed:.1f}s")
    print(f"📊 {len(final_database)} users, {total_images} ảnh")
    input("\nEnter...")

# TRAINING FACENET

def train_model_incremental():
    print_header()
    print("TRAINING NGƯỜI MỚI (INCREMENTAL)     ")

    if not os.path.exists("trainer/facenet_database.npy"):
        print(" Chưa có database cũ!")
        print(" Dùng [2] để train toàn bộ lần đầu")
        input()
        return

    # Load database cũ
    print("📂 Load database cũ...")
    old_database = np.load("trainer/facenet_database.npy", allow_pickle=True).item()
    print(f"✅ Database cũ: {len(old_database)} users")

    # Tìm tất cả user_ids trong Mauanh
    print("\n Quét thư mục Mauanh...")
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

    # Tìm users mới (chưa có trong database)
    new_user_ids = all_user_ids - set(old_database.keys())

    if not new_user_ids:
        print("\n✅ Không có user mới!")
        print("💡 Tất cả users đã được train")
        input()
        return

    print(f"✅ Tìm thấy {len(all_user_ids)} users trong Mauanh")
    print(f"🆕 Users mới: {sorted(new_user_ids)}\n")

    # Train chỉ người mới
    embeddings_dict = {}
    total_new_images = 0

    # Đếm ảnh của người mới
    for folder_name in os.listdir("Mauanh"):
        folder_path = os.path.join("Mauanh", folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("User.") and filename.endswith(".jpg"):
                    user_id = int(filename.split(".")[1])
                    if user_id in new_user_ids:
                        total_new_images += 1

        elif folder_name.startswith("User.") and folder_name.endswith(".jpg"):
            user_id = int(folder_name.split(".")[1])
            if user_id in new_user_ids:
                total_new_images += 1

    print(f"📊 Tìm thấy {total_new_images} ảnh của người mới\n")

    processed = 0
    start_time = time.time()

    # Process chỉ ảnh của người mới
    for folder_name in os.listdir("Mauanh"):
        folder_path = os.path.join("Mauanh", folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith("User.") and filename.endswith(".jpg"):
                    user_id = int(filename.split(".")[1])

                    if user_id not in new_user_ids:
                        continue  # Skip người đã có

                    path = os.path.join(folder_path, filename)
                    img = cv2.imread(path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    embedding = get_embedding(img_rgb)

                    if embedding is not None:
                        if user_id not in embeddings_dict:
                            embeddings_dict[user_id] = []
                        embeddings_dict[user_id].append(embedding)

                    processed += 1
                    print(f"\r🔄 Processing: {processed}/{total_new_images} ({processed * 100 // total_new_images}%)", end="")

        elif folder_name.startswith("User.") and folder_name.endswith(".jpg"):
            user_id = int(folder_name.split(".")[1])

            if user_id not in new_user_ids:
                continue  # Skip người đã có

            path = os.path.join("Mauanh", folder_name)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(img_rgb)

            if embedding is not None:
                if user_id not in embeddings_dict:
                    embeddings_dict[user_id] = []
                embeddings_dict[user_id].append(embedding)

            processed += 1
            print(f"\r🔄 Processing: {processed}/{total_new_images} ({processed * 100 // total_new_images}%)", end="")

    print("\n\n📊 Tính embeddings cho người mới...")

    # Tính mean embedding cho người mới
    new_embeddings = {}
    for user_id, embeddings in embeddings_dict.items():
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        new_embeddings[user_id] = mean_embedding
        print(f"   User {user_id}: {len(embeddings)} ảnh → 1 embedding")

    # Merge với database cũ
    final_database = {**old_database, **new_embeddings}

    # Save
    np.save("trainer/facenet_database.npy", np.array(final_database, dtype=object))

    elapsed = time.time() - start_time
    print(f"\n✅ Xong! File: trainer/facenet_database.npy")
    print(f"⏱️  Thời gian: {elapsed:.1f}s")
    print(f"📊 Database: {len(old_database)} cũ + {len(new_embeddings)} mới = {len(final_database)} total")
    print(f"\n💡 Tiết kiệm: Chỉ train {total_new_images} ảnh thay vì toàn bộ!")
    input("\nEnter...")

# RECOGNITION (FACENET)


def recognition_dual_camera():
    print_header()
    print("        NHẬN DIỆN (FaceNet)                  ")


    if not os.path.exists("trainer/facenet_database.npy"):
        print("❌ Chưa train! Chọn [2] hoặc [3]")
        input()
        return

    print("🔄 Load FaceNet database...")
    database = np.load("trainer/facenet_database.npy", allow_pickle=True).item()
    print(f"✅ Loaded {len(database)} users")

    users = load_users_from_google_sheets()

    if not users:
        users = {}
        for user_id in database.keys():
            users[user_id] = {'name': f'User {user_id}', 'rfid': ''}

    print(f"✅ {len(users)} người")
    print(f"🎯 MediaPipe + FaceNet")
    print(f"🎯 Threshold: {FACENET_THRESHOLD}")
    print("🟢 START - Q thoát\n")

    stop_event = threading.Event()

    def camera_thread(cam_url, cam_type, is_checkout):
        cap = cv2.VideoCapture(cam_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"❌ Camera {cam_type} lỗi!")
            return

        detector = mp_face_detection.FaceDetection(model_selection=0,
                                                   min_detection_confidence=MEDIAPIPE_CONFIDENCE)  # type: ignore
        last_id, last_time, count = -1, 0, 0
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

            if len(faces) == 0:
                status_text = "No face detected"
                status_color = (128, 128, 128)
                cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            else:
                for (x, y, w, h) in faces:
                    time_since_last = ct - last_time
                    in_cooldown = (last_id != -1 and time_since_last <= COOLDOWN_SECONDS)

                    if in_cooldown:
                        name = users[last_id]['name'] if last_id in users else "Unknown"
                        cooldown_remaining = COOLDOWN_SECONDS - time_since_last
                        col = (0, 255, 0)
                        conf_text = f"{cooldown_remaining:.0f}s"

                        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
                        cv2.putText(frame, f"{name} ({conf_text})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                        continue

                    face_bgr = frame[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                    embedding = get_embedding(face_rgb)

                    if embedding is None:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        cv2.putText(frame, "Processing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        continue

                    min_distance = float('inf')
                    predicted_id = -1

                    for user_id, db_embedding in database.items():
                        distance = compare_embeddings(embedding, db_embedding)
                        if distance < min_distance:
                            min_distance = distance
                            predicted_id = user_id

                    if min_distance < FACENET_THRESHOLD and predicted_id in users:
                        name = users[predicted_id]['name']
                        conf_text = f"{(1 - min_distance) * 100:.0f}%"

                        send_face_to_blynk(predicted_id, name, is_checkout)
                        last_id, last_time, count = predicted_id, ct, count + 1
                        col = (0, 255, 0)
                    else:
                        name = "Unknown"
                        col = (0, 0, 255)
                        conf_text = f"{(1 - min_distance) * 100:.0f}%" if min_distance < 1 else "0%"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
                    cv2.putText(frame, f"{name} ({conf_text})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            cv2.putText(frame, f"{cam_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(f'Camera {cam_type}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()

        detector.close()
        cap.release()

    t1 = threading.Thread(target=camera_thread, args=(ESP32CAM_IN, "IN", False))
    t2 = threading.Thread(target=camera_thread, args=(ESP32CAM_OUT, "OUT", True))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    cv2.destroyAllWindows()
    print("\n🔴 Tắt")
    input()

# MANAGE

def manage_users():
    print_header()
    print(" QUẢN LÝ                  ")


    users = load_users_from_google_sheets()
    if not users:
        print("❌ Không load được!")
        input()
        return

    print("📋 DANH SÁCH:\n")
    print(f"{'ID':<5} {'Tên':<25} {'RFID':<15} {'Ảnh':<10}")
    print("-" * 60)

    for uid, info in users.items():
        folder_name = f"Mauanh/{info['name'].replace(' ', '_')}"
        ic = 0

        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            ic = len([fn for fn in os.listdir(folder_name) if fn.startswith(f"User.{uid}.")])

        if os.path.exists("Mauanh"):
            ic += len([fn for fn in os.listdir("Mauanh")
                       if fn.startswith(f"User.{uid}.") and os.path.isfile(os.path.join("Mauanh", fn))])

        print(f"{uid:<5} {info['name']:<25} {info['rfid']:<15} {ic} ảnh")

    print("\n1. Xem ảnh | 2. Xóa | 0. Quay lại")
    choice = input("\nChọn: ").strip()

    if choice == '1':
        try:
            uid = int(input("\nID: ").strip())
            if uid in users:
                imgs = []
                folder_name = f"Mauanh/{users[uid]['name'].replace(' ', '_')}"

                if os.path.exists(folder_name) and os.path.isdir(folder_name):
                    imgs += [os.path.join(folder_name, fn) for fn in os.listdir(folder_name)
                             if fn.startswith(f"User.{uid}.")]

                if os.path.exists("Mauanh"):
                    imgs += [os.path.join("Mauanh", fn) for fn in os.listdir("Mauanh")
                             if fn.startswith(f"User.{uid}.") and os.path.isfile(os.path.join("Mauanh", fn))]

                if imgs:
                    idx = 0
                    while True:
                        img = cv2.imread(imgs[idx])
                        if img is not None:
                            img = cv2.resize(img, (400, 400))
                            cv2.putText(img, f"{users[uid]['name']} ({idx + 1}/{len(imgs)})",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.imshow('Images', img)
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord(' '):
                            idx = (idx + 1) % len(imgs)
                    cv2.destroyAllWindows()
        except:
            pass
        input()

    elif choice == '2':
        try:
            uid = int(input("\nID xóa: ").strip())
            if uid in users and input(f"Xóa '{users[uid]['name']}'? (yes/no): ").lower() == 'yes':
                dc = 0
                folder_name = f"Mauanh/{users[uid]['name'].replace(' ', '_')}"

                if os.path.exists(folder_name) and os.path.isdir(folder_name):
                    import shutil
                    shutil.rmtree(folder_name)
                    print(f"✅ Xóa folder: {folder_name}")

                if os.path.exists("Mauanh"):
                    for fn in os.listdir("Mauanh"):
                        if fn.startswith(f"User.{uid}.") and os.path.isfile(os.path.join("Mauanh", fn)):
                            os.remove(os.path.join("Mauanh", fn))
                            dc += 1

                print(f"✅ Xóa {dc} ảnh. Train lại!")
        except:
            pass
        input()

# MAIN
def main():
    while True:
        print_menu()
        choice = input("\nChọn: ").strip()
        if choice == '1':
            collect_faces()
        elif choice == '2':
            train_model()
        elif choice == '3':
            train_model_incremental()
        elif choice == '4':
            recognition_dual_camera()
        elif choice == '5':
            manage_users()
        elif choice == '0':
            print("\n👋 Bye!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Stop!")
    finally:
        cv2.destroyAllWindows()