import cv2
import numpy as np
import math
from ultralytics import YOLO

# --- 1. CẤU HÌNH ---
MODEL_PATH = r'best.pt'
IMG_PATH = r'D:\NCKHH\anhthi\anh5.jpg' # Thay bằng đường dẫn ảnh của bạn

ANSWER_KEY = {
    "025": {i: v for i, v in zip(range(1, 51), ['A','B','C','D','A','B','C','D','A','B']*5)}
}

CONFIG_RATIOS = {
    'MADE':    {'y_top': 0.08,  'y_bot': 0.09, 'x_start': 0.08, 'x_step': 0.08},
    'KHOI_17': {'y_top': 0.055, 'y_bot': 0.05, 'x_start': 0.05, 'x_step': 0.045},
    'KHOI_16': {'y_top': 0.055, 'y_bot': 0.11, 'x_start': 0.05, 'x_step': 0.045}
}

# --- 2. HÀM TỐI ƯU CĂN CHỈNH ---

def order_points(pts):
    """Sắp xếp 4 tọa độ theo thứ tự: TL, TR, BR, BL"""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def get_markers_robust(img_gray):
    """Tìm mốc đen thông minh, chịu được bóng râm và khoảng cách xa gần"""
    # 1. Adaptive Threshold xử lý ánh sáng không đều
    thresh = cv2.adaptiveThreshold(img_gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    # 2. Phép toán hình thái học để nối các nét đứt của mốc đen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tính diện tích tương đối (0.05% đến 1.5% diện tích ảnh)
    h, w = img_gray.shape
    img_area = h * w
    min_a, max_a = img_area * 0.0005, img_area * 0.015

    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if min_a < area < max_a:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            # Chấp nhận các khối có 4-6 cạnh để bù sai số
            if 4 <= len(approx) <= 6:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    
    # Nếu tìm thấy nhiều hơn 4 mốc, lấy 4 mốc xa nhau nhất (ở 4 góc)
    if len(centers) > 4:
        # Sắp xếp theo khoảng cách tới trung tâm ảnh hoặc lấy các cực trị
        centers = sorted(centers, key=lambda p: p[0]+p[1])
        return [centers[0], centers[-1], 
                min(centers, key=lambda p: p[1]-p[0]), 
                max(centers, key=lambda p: p[1]-p[0])]
    return centers

def get_grid_points(m, rows, cols, start_q):
    tx, ty = m[0]; bx, by = m[1]; dist_y = by - ty
    cfg = CONFIG_RATIOS['MADE'] if rows == 10 else (CONFIG_RATIOS['KHOI_16'] if rows == 16 else CONFIG_RATIOS['KHOI_17'])
    row_h = (dist_y * (1 - cfg['y_top'] - cfg['y_bot'])) / (rows - 1)
    col_w = dist_y * cfg['x_step']
    s_x, s_y = tx + (dist_y * cfg['x_start']), ty + (dist_y * cfg['y_top'])
    grid = []
    choices = ['A', 'B', 'C', 'D']
    for r in range(rows):
        for c in range(cols):
            grid.append({'pos': (int(s_x + c * col_w), int(s_y + r * row_h)), 'q': start_q + r, 'choice': choices[c] if rows != 10 else str(r), 'col': c})
    return grid, row_h

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
model = YOLO(MODEL_PATH)
img = cv2.imread(IMG_PATH)
if img is None: print("Lỗi đường dẫn!"); exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Sử dụng hàm tìm mốc mới
m_full = get_markers_robust(gray)

if len(m_full) >= 4:
    # Sắp xếp 4 góc chuẩn xác trước khi Warp
    rect = order_points(m_full[:4])
    M_matrix = cv2.getPerspectiveTransform(rect, np.array([[0,0],[800,0],[800,1400],[0,1400]], dtype="float32"))
    
    warped_img = cv2.warpPerspective(img, M_matrix, (800, 1400))
    # Tạo thres riêng cho ảnh đã bẻ thẳng để tìm marker cục bộ dễ hơn
    warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    warped_thres = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    out_img = warped_img.copy()

    roi_cfgs = [
        {'box': [0, 380, 630, 750], 'r': 10, 'c': 3, 's': 0, 'tag': 'MADE'},
        {'box': [635, 1400, 50, 290], 'r': 17, 'c': 4, 's': 1, 'tag': 'K1'},
        {'box': [635, 1400, 290, 480], 'r': 17, 'c': 4, 's': 18, 'tag': 'K2'},
        {'box': [635, 1400, 480, 680], 'r': 16, 'c': 4, 's': 35, 'tag': 'K3'}
    ]

    roi_imgs = [warped_img[c['box'][0]:c['box'][1], c['box'][2]:c['box'][3]] for c in roi_cfgs]
    y_results = model.predict(roi_imgs, conf=0.5, verbose=False)

    user_ans, made, processed_data = {}, {0:"X", 1:"X", 2:"X"}, []

    for i, cfg in enumerate(roi_cfgs):
        y1, y2, x1, x2 = cfg['box']
        crop_gray = warped_gray[y1:y2, x1:x2]
        # Tìm marker cục bộ bằng diện tích nhỏ hơn
        m_roi = sorted(get_markers_robust(crop_gray), key=lambda x: x[1])[:2]
        
        if len(m_roi) >= 2:
            grid, rh = get_grid_points(m_roi, cfg['r'], cfg['c'], cfg['s'])
            y_pts = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in y_results[i].boxes.xyxy]
            for py in y_pts:
                match = min(grid, key=lambda pg: math.sqrt((py[0]-pg['pos'][0])**2 + (py[1]-pg['pos'][1])**2))
                if math.sqrt((py[0]-match['pos'][0])**2 + (py[1]-match['pos'][1])**2) < rh*0.7:
                    if cfg['tag'] == 'MADE': made[match['col']] = match['choice']
                    else: user_ans[match['q']] = match['choice']
            processed_data.append((cfg, grid))

    # --- CHẤM ĐIỂM ---
    ma_de = f"{made[0]}{made[1]}{made[2]}"
    correct, key = 0, ANSWER_KEY.get(ma_de, {})
    for cfg, grid in processed_data:
        if cfg['tag'] == 'MADE': continue
        y1, y2, x1, x2 = cfg['box']
        for p in grid:
            u_c, k_c = user_ans.get(p['q']), key.get(p['q'])
            px, py = x1 + p['pos'][0], y1 + p['pos'][1]
            if p['choice'] == k_c:
                cv2.circle(out_img, (px, py), 12, (0, 255, 0), 2)
                if u_c == k_c: correct += 1
            if u_c == p['choice'] and u_c != k_c:
                cv2.line(out_img, (px-10, py-10), (px+10, py+10), (0, 0, 255), 2)
                cv2.line(out_img, (px+10, py-10), (px-10, py+10), (0, 0, 255), 2)

    score = (correct / 50) * 10
    cv2.rectangle(out_img, (0,0), (800, 80), (255,255,255), -1)
    cv2.putText(out_img, f"MADE: {ma_de} | SCORE: {score:.2f}", (20, 50), 2, 1, (0, 0, 255), 2)
    cv2.imshow("Kq NCKH Optimized", cv2.resize(out_img, (550, 850)))
    cv2.waitKey(0)
else:
    print("Không tìm thấy đủ 4 mốc góc!")