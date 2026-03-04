import cv2
import numpy as np
import math
import base64
import os
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO

# --- 1. CẤU HÌNH SERVER & MODEL ---
app = FastAPI(title="Hệ thống Chấm thi NCKH - Smart Alignment", version="3.0")

print("Đang tải model YOLO...")
MODEL_PATH = r'best.pt'
model = YOLO(MODEL_PATH) 
print("Tải model thành công!")

ANSWER_KEY = {
    "025": {i: v for i, v in zip(range(1, 51), ['A','B','C','D','A','B','C','D','A','B']*5)}
}

CONFIG_RATIOS = {
    'MADE':    {'y_top': 0.08,  'y_bot': 0.09, 'x_start': 0.08, 'x_step': 0.08},
    'KHOI_17': {'y_top': 0.055, 'y_bot': 0.05, 'x_start': 0.05, 'x_step': 0.045},
    'KHOI_16': {'y_top': 0.055, 'y_bot': 0.11, 'x_start': 0.05, 'x_step': 0.045}
}

# --- 2. THUẬT TOÁN CĂN CHỈNH THÔNG MINH ---

def order_points(pts):
    """Sắp xếp 4 tọa độ theo thứ tự chuẩn: Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái"""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_markers_robust(img_gray):
    """Tìm mốc đen dựa trên tỷ lệ diện tích ảnh và Adaptive Threshold"""
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_gray.shape
    img_area = h * w
    min_a, max_a = img_area * 0.0005, img_area * 0.015 # Tỷ lệ % diện tích

    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if min_a < area < max_a:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if 4 <= len(approx) <= 6:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
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

# --- 3. HÀM LÕI CHẤM THI ---

def core_cham_thi(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m_full = get_markers_robust(gray)

        if len(m_full) < 4:
            return False, "", 0, 0, "Không tìm đủ 4 góc định vị"

        # Warp Perspective với tọa độ đã sắp xếp chuẩn
        rect = order_points(m_full[:4])
        M_matrix = cv2.getPerspectiveTransform(rect, np.array([[0,0],[800,0],[800,1400],[0,1400]], dtype="float32"))
        warped_img = cv2.warpPerspective(img, M_matrix, (800, 1400))
        
        # Tiền xử lý ảnh đã bẻ thẳng
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        
        roi_cfgs = [
            {'box': [0, 380, 630, 750], 'r': 10, 'c': 3, 's': 0, 'tag': 'MADE'},
            {'box': [635, 1400, 50, 290], 'r': 17, 'c': 4, 's': 1, 'tag': 'K1'},
            {'box': [635, 1400, 290, 480], 'r': 17, 'c': 4, 's': 18, 'tag': 'K2'},
            {'box': [635, 1400, 480, 680], 'r': 16, 'c': 4, 's': 35, 'tag': 'K3'}
        ]

        roi_imgs = [warped_img[c['box'][0]:c['box'][1], c['box'][2]:c['box'][3]] for c in roi_cfgs]
        y_results = model.predict(roi_imgs, conf=0.5, verbose=False)

        user_ans = {}
        made = {0:"X", 1:"X", 2:"X"}

        for i, cfg in enumerate(roi_cfgs):
            y1, y2, x1, x2 = cfg['box']
            crop_gray = warped_gray[y1:y2, x1:x2]
            m_roi = sorted(get_markers_robust(crop_gray), key=lambda x: x[1])[:2]
            
            if len(m_roi) >= 2:
                grid, rh = get_grid_points(m_roi, cfg['r'], cfg['c'], cfg['s'])
                y_pts = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in y_results[i].boxes.xyxy]
                
                for py in y_pts:
                    match = min(grid, key=lambda pg: math.sqrt((py[0]-pg['pos'][0])**2 + (py[1]-pg['pos'][1])**2))
                    if math.sqrt((py[0]-match['pos'][0])**2 + (py[1]-match['pos'][1])**2) < rh*0.7:
                        if cfg['tag'] == 'MADE': made[match['col']] = match['choice']
                        else: user_ans[match['q']] = match['choice']

        ma_de_str = f"{made[0]}{made[1]}{made[2]}"
        correct = 0
        key = ANSWER_KEY.get(ma_de_str, {})
        for q, ans in user_ans.items():
            if key.get(q) == ans: correct += 1

        return True, ma_de_str, correct, (correct/50)*10, "Thành công"
    except Exception as e:
        return False, "", 0, 0, str(e)

# --- 4. CÁC API ENDPOINT ---

@app.post("/api/v1/cham-thi")
async def cham_thi_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    success, ma_de, correct, score, msg = core_cham_thi(img)
    if success:
        return {"status": "success", "data": {"ma_de": ma_de, "so_cau_dung": correct, "diem": float(score)}}
    return JSONResponse(status_code=400, content={"status": "error", "message": msg})

@app.post("/api/v1/cham-thu-muc")
async def cham_thu_muc_api(folder_path: str = Form(...)):
    if not os.path.exists(folder_path):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Thư mục không tồn tại"})

    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                success, ma_de, correct, score, msg = core_cham_thi(img)
                results.append({
                    "Tên File": filename, "Mã đề": ma_de if success else "-",
                    "Câu đúng": correct, "Điểm": round(score, 2), "Ghi chú": msg
                })

    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(folder_path, "KetQua_TuDong_V3.xlsx")
        df.to_excel(output_path, index=False)
        return {"status": "success", "excel_saved_at": output_path, "data": results}
    return JSONResponse(status_code=400, content={"status": "error", "message": "Không tìm thấy ảnh"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)