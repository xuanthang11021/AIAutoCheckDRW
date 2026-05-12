from __future__ import annotations
import cv2
from matplotlib.pyplot import box
import numpy as np
import os
import ctypes
from pathlib import Path
import re
import json
from datetime import datetime
from paddleocr import PaddleOCR
import gc

# ============================================================================
# CẤU HÌNH HỆ THỐNG & SỬA LỖI THƯ VIỆN
# ============================================================================

try:
    anaconda_lib = "/home/tim-advance-user/anaconda3/envs/autoCheckFA_py39/lib"
    if os.path.exists(anaconda_lib):
        os.environ["LD_LIBRARY_PATH"] = anaconda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["FLAGS_use_onednn"] = "0"
    os.environ["DNNL_VERBOSE"] = "0"
    for lib in ["libiomp5.so", "libmkl_rt.so", "libmkl_core.so"]:
        lib_path = os.path.join(anaconda_lib, lib)
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    print("✅ Cấu hình thư viện MKL thành công.")
except Exception as e:
    print(f"⚠️ Cảnh báo cấu hình thư viện: {e}")

OCR_TYPE = os.getenv("OCR_TYPE", "PADDLE")

# ============================================================================
# PHẦN 1: PHÁT HIỆN VÀ TRÍCH XUẤT VÙNG
# ============================================================================

class ImageRegionExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.image is None: raise ValueError(f"Could not read image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def group_nearby_boxes(self, boxes, horizontal_threshold=60, vertical_threshold=40):
        if not boxes: return []
        boxes = np.array(boxes)
        n = len(boxes)
        merged = np.zeros(n, dtype=bool)
        grouped_boxes = []
        for i in range(n):
            if merged[i]: continue
            current_group_indices = [i]
            merged[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if merged[j]: continue
                    should_merge = False
                    for idx in current_group_indices:
                        x1, y1, w1, h1 = boxes[idx]
                        x2, y2, w2, h2 = boxes[j]
                        h_dist = max(0, x2 - (x1 + w1), (x1 - (x2 + w2)))
                        v_dist = max(0, y2 - (y1 + h1), (y1 - (y2 + h2)))
                        if h_dist <= horizontal_threshold and v_dist <= vertical_threshold:
                            should_merge = True; break
                    if should_merge:
                        current_group_indices.append(j); merged[j] = True; changed = True
            gb = boxes[current_group_indices]
            x_min, y_min = np.min(gb[:, 0]), np.min(gb[:, 1])
            x_max, y_max = np.max(gb[:, 0] + gb[:, 2]), np.max(gb[:, 1] + gb[:, 3])
            grouped_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        return grouped_boxes

    def _find_small_contours(self, binary_image):
        boxes = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 3 < w < 600 and 3 < h < 120: boxes.append((x, y, w, h))
        return boxes

    def _remove_duplicates(self, boxes, iou_threshold=0.5):
        if not boxes: return []
        boxes_np = np.array(boxes)
        x1, y1, x2, y2 = boxes_np[:,0], boxes_np[:,1], boxes_np[:,0]+boxes_np[:,2], boxes_np[:,1]+boxes_np[:,3]
        areas = (x2-x1)*(y2-y1)
        order = areas.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1, yy1, xx2, yy2 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]]), np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0, xx2-xx1), np.maximum(0, yy2-yy1)
            iou = (w*h) / (areas[i] + areas[order[1:]] - (w*h) + 1e-6)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return [boxes[i] for i in keep]

    def detect_regions(self, min_width=30, min_height=15):
        all_small = []
        _, b1 = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        all_small.extend(self._find_small_contours(b1))
        b2 = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        all_small.extend(self._find_small_contours(b2))
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(self.gray)
        for r in regions:
            x, y, w, h = cv2.boundingRect(r)
            if 3 < w < 600 and 3 < h < 120: all_small.append((x, y, w, h))
        all_small = self._remove_duplicates(all_small)
        grouped = self.group_nearby_boxes(all_small)
        return [r for r in grouped if r[2] >= min_width and r[3] >= min_height]

    def extract_and_save_regions(self, bboxes, output_dir='extracted_regions', padding_horizontal=15, padding_vertical=20, padding_top_extra=10):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        saved = []
        for i, (x, y, w, h) in enumerate(bboxes, 1):
            y1, y2 = max(0, y-padding_vertical-padding_top_extra), min(self.image.shape[0], y+h+padding_vertical)
            x1, x2 = max(0, x-padding_horizontal), min(self.image.shape[1], x+w+padding_horizontal)
            roi = self.image[y1:y2, x1:x2]
            if roi.size == 0: continue
            if h > w * 1.2: roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            fn = f"{output_dir}/region_{i:03d}.png"
            cv2.imwrite(fn, roi); saved.append(fn)
        return saved

    def visualize_regions(self, bboxes, output_path='detected_regions.jpg'):
        vis = self.image.copy()
        for i, (x, y, w, h) in enumerate(bboxes, 1):
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, f"#{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(output_path, vis)

# ============================================================================
# PHẦN 2: OCR VÀ XỬ LÝ VĂN BẢN (CIRCLE-FIRST)
# ============================================================================

OCR_INSTANCE = None

def get_ocr_instance():
    global OCR_INSTANCE
    if OCR_INSTANCE is None:
        if OCR_TYPE == "GLM":
            from glm_ocr_handler import get_glm_ocr
            OCR_INSTANCE = get_glm_ocr()
        else:
            OCR_INSTANCE = PaddleOCR(use_textline_orientation=False, lang='en', text_det_limit_side_len=4000)
    return OCR_INSTANCE

def reset_ocr_instance():
    global OCR_INSTANCE
    OCR_INSTANCE = None

def format_ocr_result(raw):
    if not raw: return []
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list) and len(raw[0]) > 0 and isinstance(raw[0][0], list): return raw
    formatted = []
    results = raw if isinstance(raw, list) else [raw]
    for res in results:
        if hasattr(res, 'json'):
            formatted.append([[item['dt_polys'], [item['rec_text'], item['rec_score']]] for item in res.json.get('ocr_res', [])])
        elif isinstance(res, list): formatted.append(res)
        else: formatted.append([])
    return formatted

def preprocess_image_region(image_path, scale=5):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None, None
    best_b, best_r = None, None
    for s in [scale, 4, 3]:
        resized = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(cv2.bilateralFilter(resized, 9, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if 0.1 < (np.sum(binary == 255) / binary.size) < 0.85:
            best_b, best_r = binary, resized; break
    if best_b is None:
        best_r = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, best_b = cv2.threshold(cv2.bilateralFilter(best_r, 9, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tmp_path = f"{image_path}_pre.png"; cv2.imwrite(tmp_path, best_b)
    return tmp_path, best_b, best_r

def _fix_misread_deviation(t):
    return t[1:] if re.match(r'^<[+\-]\d+\.?\d*$', t) else t

def _strip_decorators(t):
    t = re.sub(r'[★☆✦✧✱•·\*]', '', t)
    for kw in ['WARPAGE', 'ARPAGE', 'MAX', 'MIN', 'TYP', 'REF', 'TOTAL']:
        t = re.sub(re.escape(kw), '', t, flags=re.IGNORECASE)
    return t.strip(': ').strip()

def detect_circles_around_text(img, items):
    circles = cv2.HoughCircles(cv2.GaussianBlur(img, (9,9), 2), cv2.HOUGH_GRADIENT, 1, 40, param1=80, param2=35, minRadius=20, maxRadius=55)
    circled = set()
    if circles is not None:
        for (cx, cy, r) in np.round(circles[0, :]).astype("int"):
            for it in items:
                if np.sqrt((it['cx']-cx)**2 + (it['cy']-cy)**2) < r * 0.7 and re.match(r'^\d+$', it['text']): circled.add(id(it))
    return circled

def get_item_number_from_circle(img):
    """Specifically looks for a circle and extracts the number inside it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 40, param1=80, param2=35, minRadius=20, maxRadius=60)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda c: c[0]) 
        for (cx, cy, r) in circles:
            y1, y2 = max(0, cy-r), min(img.shape[0], cy+r)
            x1, x2 = max(0, cx-r), min(img.shape[1], cx+r)
            circle_crop = img[y1:y2, x1:x2]
            if circle_crop.size > 0:
                ocr = get_ocr_instance()
                res = format_ocr_result(ocr.ocr(circle_crop))
                if res and res[0]:
                    for line in res[0]:
                        text = re.sub(r'\D', '', line[1][0])
                        if text and 1 <= len(text) <= 4:
                            print(f"  [CIRCLE-FIRST] Found Item No: ({text})")
                            return text
    return None

def _ocr_multiregion(path, ocr):
    img = cv2.imread(str(path))
    if img is None: return []
    h, w = img.shape[:2]
    scale = max(1, 1000 // max(h, w))
    if scale > 1: img = cv2.resize(img, None, fx=scale, fy=scale)
    h, w = img.shape[:2]
    all_items, seen = [], set()
    for r in [(0,0,w,h), (0,0,w,h*2//3), (0,h//3,w,h), (0,0,w//2,h), (w//3,0,w,h)]:
        crop = img[r[1]:r[3], r[0]:r[2]]
        if crop.size == 0: continue
        for p_img in [crop, cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]:
            try:
                res = format_ocr_result(ocr.ocr(p_img))
                if res and res[0]:
                    for line in res[0]:
                        t = line[1][0].strip()
                        if t and t not in seen:
                            box = line[0]; seen.add(t)
                            all_items.append({'text':t, 'conf':line[1][1], 'box':box, 'cx':(box[0][0]+box[2][0])/2, 'cy':(box[0][1]+box[2][1])/2})
            except Exception as e: print(f"    [MULTIREGION Error] {e}")
    return all_items

def deduplicate_fragments(frags):
    frags.sort(key=lambda x: x.get('conf', 0), reverse=True)
    keep = []
    for f1 in frags:
        dup = False
        for f2 in keep:
            x1, y1, x2, y2 = min(p[0] for p in f1['box']), min(p[1] for p in f1['box']), max(p[0] for p in f1['box']), max(p[1] for p in f1['box'])
            ax1, ay1, ax2, ay2 = min(p[0] for p in f2['box']), min(p[1] for p in f2['box']), max(p[0] for p in f2['box']), max(p[1] for p in f2['box'])
            inter = max(0, min(x2, ax2) - max(x1, ax1)) * max(0, min(y2, ay2) - max(y1, ay1))
            if inter / ( (x2-x1)*(y2-y1) + 1e-6 ) > 0.6 and (f1['text'] in f2['text'] or f2['text'] in f1['text']):
                dup = True; break
        if not dup: keep.append(f1)
    return keep

def smart_cleanup_and_process_items(items, img, circled_ids):
    processed = []
    for it in items:
        t = _fix_misread_deviation(_strip_decorators(it['text']))
        if re.match(r'^\d{5,}\.\d+$', t):
            processed.append({**it, 'text': t[:3], 'cx': it['cx']-20, 'is_circled': True})
            processed.append({**it, 'text': t[3:], 'cx': it['cx']+20, 'is_circled': False})
        else:
            it['text'] = t
            it['is_circled'] = id(it) in circled_ids or (re.match(r'^\d+$', t) and int(t) < 1000 and it['cx'] < 150)
            processed.append(it)
    final = []
    for it in processed:
        t = it['text']
        if not t or t in ['*', '×', '★']: continue
        if re.search(r'[+\-±]\d', t):
            if t.startswith('<') and not t.endswith('>'): t += '>'
            if t.endswith('>') and not t.startswith('<'): t = '<' + t
        if re.match(r'^<[+\-±]?\d+\.?\d*>$', t):
            m = re.search(r'([+\-±]?\d+\.?\d*)', t)
            if m: t = f"<±{m.group(1).lstrip('+-±')}>"
        it['text'] = t; final.append(it)
    return final

def parse_dimension_structure(frags, forced_item_no=None):
    refs, noms, devs, tols = [], [], [], []
    if forced_item_no: refs.append({'text': forced_item_no, 'is_circled': True, 'box': [[0,0],[0,0],[0,0],[0,0]]})
    print(f"  [DEBUG-PARSE] Fragments: {[f['text'] for f in frags]}")
    for f in frags:
        t = f['text'].strip()
        if not t: continue
        if forced_item_no and t == forced_item_no:
            if not any(r['text'] == forced_item_no for r in refs): refs.append(f)
            continue
        combined_match = re.match(r'^(\d+\.?\d*)<([+\-±]?\d+\.?\d*)>$', t)
        if combined_match:
            noms.append({**f, 'text': combined_match.group(1)})
            tols.append({**f, 'text': f"<±{combined_match.group(2).lstrip('+-±')}>"})
            continue
        fused_dev_match = re.match(r'^(\d+\.?\d*)([+\-]\d+\.?\d*)$', t)
        if fused_dev_match:
            noms.append({**f, 'text': fused_dev_match.group(1)})
            devs.append({**f, 'text': fused_dev_match.group(2)})
            continue
        if f.get('is_circled') and re.match(r'^\d+$', t):
            if not forced_item_no: refs.append(f)
        elif (t.startswith('<') and t.endswith('>')) or '±' in t: tols.append(f)
        elif t.startswith('+') or t.startswith('-'): devs.append(f)
        elif re.match(r'^\d+(\.\d+)?$', t): noms.append(f)
    print(f"  [DEBUG-PARSE] Categorized: refs={[r['text'] for r in refs]}, noms={[n['text'] for n in noms]}, devs={[d['text'] for d in devs]}, tols={[t['text'] for t in tols]}")
    res = []
    if refs:
        unique_refs, texts = [], [r['text'] for r in refs]
        for r in refs:
            if not any(r['text'] != other and r['text'] in other for other in texts): unique_refs.append(r)
        if not unique_refs: unique_refs = refs
        unique_refs.sort(key=lambda x: len(x['text']), reverse=True); res.append(f"({unique_refs[0]['text']})")
    if noms:
        if forced_item_no: noms = [n for n in noms if n['text'] != forced_item_no]
        if noms:
            noms.sort(key=lambda f: (max(p[0] for p in f['box'])-min(p[0] for p in f['box']))*(max(p[1] for p in f['box'])-min(p[1] for p in f['box'])), reverse=True); res.append(noms[0]['text'])
    up, lo = None, None
    if devs:
        pos, neg = [d for d in devs if d['text'].startswith('+')], [d for d in devs if d['text'].startswith('-')]
        up, lo = (pos[0]['text'] if pos else None), (neg[0]['text'] if neg else None)
        if len(devs) >= 2 and (up is None or lo is None):
            devs.sort(key=lambda d: d['cy']); up, lo = devs[0]['text'], devs[1]['text']
    if up and lo:
        try:
            if float(re.sub(r'[^\d.\-]', '', up)) < float(re.sub(r'[^\d.\-]', '', lo)): up, lo = lo, up
        except: pass
        res.append(f"({up}/{lo})")
    elif up or lo: res.append(f"({up or lo})")
    if tols:
        tols.sort(key=lambda f: (0 if '±' in f['text'] else 1, len(f['text'])))
        m = re.search(r'([+\-±]?\d+\.?\d*)', tols[0]['text'])
        if m: res.append(f"<±{m.group(1).lstrip('+-±')}>")
    final_str = ' '.join(res)
    print(f"  [DEBUG-PARSE] Final: '{final_str}'")
    return final_str

def _count_useful_from_items(items):
    texts = [it['text'] for it in items]
    has_nom = any(re.match(r'^\d+(\.\d+)?$', t) for t in texts)
    has_dev = any(re.match(r'^<?[+\-±]\d+\.?\d*>?$', t) for t in texts)
    return (1 if has_nom else 0) + (1 if has_dev else 0)

def read_and_process_text(path, max_retries=2):
    img_orig, forced_item_no = cv2.imread(str(path)), None
    if img_orig is not None: forced_item_no = get_item_number_from_circle(img_orig)
    if OCR_TYPE == "GLM":
        try:
            h = get_ocr_instance(); img = cv2.imread(str(path))
            if img is not None and max(img.shape[:2]) < 800:
                s = 800.0 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC); raw = h.ocr(img)
            else: raw = h.ocr(str(path))
            if raw:
                clean = re.sub(r'(\d)(\()', r'\1 \2', raw.replace('$', '').replace('\\pm', '±').replace('\\', ''))
                clean = re.sub(r'(\))(\d)', r'\1 \2', clean)
                frags = []
                for i, p in enumerate(clean.split()):
                    ref = False
                    if re.match(r'^\(\d+\)$', p): p = p[1:-1]; ref = True
                    elif re.match(r'^\d+$', p) and int(p) < 1000 and (i==0 or i==len(clean.split())-1): ref = True
                    frags.append({'text':p, 'is_circled':ref, 'box':[[0,0],[100,0],[100,20],[0,20]], 'cx':0, 'cy':0})
                return parse_dimension_structure(frags, forced_item_no=forced_item_no) or raw.split('\n')[0]
            return ""
        except: return ""
    for attempt in range(max_retries):
        try:
            pre_path, pre_img, orig_res = preprocess_image_region(path)
            ocr, all_detected = get_ocr_instance(), []
            try:
                r1 = format_ocr_result(ocr.ocr(pre_path))
                if r1 and r1[0]:
                    print(f"  [PASS 1 Binary] Detected: {[(l[1][0], round(l[1][1],2)) for l in r1[0]]}")
                    for line in r1[0]:
                        b = line[0]; all_detected.append({'text':line[1][0], 'conf':line[1][1], 'box':b, 'cx':(b[0][0]+b[2][0])/2, 'cy':(b[0][1]+b[2][1])/2})
            except Exception as e: print(f"  ⚠️ Pass 1 Error: {e}")
            try:
                img_c = cv2.imread(str(path))
                if img_c is not None:
                    h, w = img_c.shape[:2]; s = 1000.0 / max(h, w)
                    r2 = format_ocr_result(ocr.ocr(cv2.resize(img_c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)))
                    if r2 and r2[0]:
                        print(f"  [PASS 2 Color] Detected: {[(l[1][0], round(l[1][1],2)) for l in r2[0]]}")
                        for line in r2[0]:
                            b = [[p[0]/s, p[1]/s] for p in line[0]]; all_detected.append({'text':line[1][0], 'conf':line[1][1], 'box':b, 'cx':(b[0][0]+b[2][0])/2, 'cy':(box[0][1]+box[2][1])/2 if 'box' in locals() else (b[0][1]+b[2][1])/2})
            except Exception as e: print(f"  ⚠️ Pass 2 Error: {e}")
            try:
                gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY); _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dilated_inv = cv2.bitwise_not(cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1))
                r3 = format_ocr_result(ocr.ocr(dilated_inv))
                if r3 and r3[0]:
                    print(f"  [PASS 3 Dilate] Detected: {[(l[1][0], round(l[1][1],2)) for l in r3[0]]}")
                    for line in r3[0]:
                        b = line[0]; all_detected.append({'text':line[1][0], 'conf':line[1][1], 'box':b, 'cx':(b[0][0]+b[2][0])/2, 'cy':(b[0][1]+b[2][1])/2})
            except: pass
            if _count_useful_from_items(all_detected) < 2:
                m_items = _ocr_multiregion(path, ocr)
                if m_items: all_detected.extend(m_items)
            if not all_detected: return ""
            all_detected = deduplicate_fragments(all_detected)
            circled_ids = detect_circles_around_text(orig_res, all_detected)
            final_frags = smart_cleanup_and_process_items(all_detected, pre_img, circled_ids)
            final_frags.sort(key=lambda x: (x['cy'] // 15, x['cx']))
            return parse_dimension_structure(final_frags, forced_item_no=forced_item_no)
        except Exception as e:
            if attempt < max_retries - 1: continue
            return ""
    return ""

def process_single_image(path, output_dir='extracted_regions', min_width=30, min_height=30, padding=10):
    res = {"image_path": str(path), "status": "processing", "regions": [], "ocr_results": []}
    try:
        ext = ImageRegionExtractor(str(path)); regions = ext.detect_regions(min_width=min_width, min_height=min_height)
        if not regions: return {"status": "no_regions_found", "image_path": str(path)}
        regions.sort(key=lambda r: (r[1], r[0])); ext.visualize_regions(regions, f"{output_dir}/detected_regions.jpg")
        cropped = ext.extract_and_save_regions(regions, output_dir=output_dir, padding_horizontal=padding, padding_vertical=padding)
        res["regions"] = cropped
        for i, crop_file in enumerate(cropped, 1):
            try:
                ocr_text = read_and_process_text(crop_file)
                res["ocr_results"].append({"region_file": crop_file, "region_index": i, "ocr_text": ocr_text, "status": "success"})
                print(f"✅ Vùng {i} OCR: {ocr_text}")
            except Exception as e: res["ocr_results"].append({"region_file": crop_file, "region_index": i, "error": str(e), "status": "error"})
        res["status"] = "success"
    except Exception as e: res["status"] = "error"; res["error"] = str(e)
    return res

def process_folder(folder, output_dir='extracted_regions', output_json='ocr_results.json', min_width=30, min_height=30, padding=10):
    p = Path(folder)
    if not p.exists(): return
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
    if not files: return
    results = {"processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "total_images": len(files), "results": []}
    for f in sorted(files): results["results"].append(process_single_image(str(f), output_dir=output_dir, min_width=min_width, min_height=min_height, padding=padding))
    with open(output_json, 'w', encoding='utf-8') as out: json.dump(results, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys; img_path = sys.argv[1] if len(sys.argv) > 1 else 'image/3.png'
    print(json.dumps(process_single_image(img_path), indent=2, ensure_ascii=False))
