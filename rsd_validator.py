import pyodbc
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RSDValidator:
    def __init__(self):
        self.server = os.getenv("DB_SERVER")
        self.database = os.getenv("DB_NAME")
        self.table = os.getenv("DB_TABLE")
        self.username = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.driver = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")
        self.connection = None
        
    def _get_connection(self):
        """Tạo kết nối tới SQL Server."""
        try:
            port = os.getenv("DB_PORT", "1433")
            conn_parts = [
                f"DRIVER={self.driver}",
                f"SERVER={self.server},{port}",
                f"DATABASE={self.database}",
                "TrustServerCertificate=yes",
                "LoginTimeout=30",
            ]
            
            if self.username and self.password:
                conn_parts.append(f"UID={self.username}")
                conn_parts.append(f"PWD={self.password}")
            else:
                conn_parts.append("Trusted_Connection=yes")
                
            conn_str = ";".join(conn_parts)
            return pyodbc.connect(conn_str)
        except Exception as e:
            print(f"❌ Error connecting to SQL Server: {e}")
            return None

    def parse_ocr_text(self, text):
        """
        Phân tích chuỗi OCR: '(206) 20.9 (+0.05/-0.184) <±0.05>' 
        -> item_no: 206, nominal: 20.9, upper: +0.05, lower: -0.184
        """
        result = {
            'item_no': None,
            'nominal': None,
            'upper': None,
            'lower': None,
            'tolerance': None,
            'full_text': text
        }
        
        # 1. Tìm Item No (số trong ngoặc đơn)
        item_match = re.search(r'\((\d+)\)', text)
        if item_match:
            result['item_no'] = item_match.group(1)
            
        # 2. Tìm Nominal (số sau ngoặc item_no)
        nominal_match = re.search(r'\)\s*(\d+\.?\d*)', text)
        if nominal_match:
            result['nominal'] = nominal_match.group(1)
            
        # 3. Tìm dung sai (dạng +X.XX / -Y.YY bên trong ngoặc đơn)
        # Hỗ trợ cả trường hợp có hoặc không có khoảng trắng
        dev_match = re.search(r'\(\s*([+-]?\d+\.?\d*)\s*/\s*([+-]?\d+\.?\d*)\s*\)', text)
        if dev_match:
            result['upper'] = dev_match.group(1)
            result['lower'] = dev_match.group(2)
        else:
            # Thử tìm dạng đơn lẻ ±X.XX hoặc <±X.XX>
            tol_match = re.search(r'[±<](\d+\.?\d*)', text)
            if tol_match:
                result['tolerance'] = tol_match.group(1)
                
        return result

    def validate(self, part_no, ocr_text):
        """
        So sánh kết quả OCR với dữ liệu từ SQL Server.
        """
        parsed = self.parse_ocr_text(ocr_text)
        if not parsed['item_no']:
            return 'not_found', 'Could not parse Item No from OCR', None
            
        conn = self._get_connection()
        if not conn:
            return 'error', 'Database connection failed', None

        try:
            cursor = conn.cursor()
            item_no = str(parsed['item_no']).strip()
            
            # Truy vấn DimensionNo chính xác hoặc đã được TRIM
            query = f"SELECT PartNo, DimensionNo, RSDValue FROM {self.table} WHERE PartNo = ? AND (DimensionNo = ? OR LTRIM(RTRIM(DimensionNo)) = ?)"
            cursor.execute(query, (part_no, item_no, item_no))
            row = cursor.fetchone()

            if not row:
                return 'not_found', f'Item {item_no} for Part {part_no} not found in Database', None
            
            # Lấy RSDValue từ cột 3 (index 2)
            db_rsd_value = str(row[2]).replace(" ", "")
            
            # LOGIC SO SÁNH THÔNG MINH
            is_match = False
            
            # 1. Trích xuất thông tin từ DB (ví dụ: 10.35(+0.050/-0.074))
            db_nominal_val, db_upper_val, db_lower_val = None, None, None
            db_match = re.search(r'^(\d+\.?\d*)\(([+-]?\d+\.?\d*)/([+-]?\d+\.?\d*)\)', db_rsd_value)
            if db_match:
                db_nominal_val = db_match.group(1)
                db_upper_val = db_match.group(2)
                db_lower_val = db_match.group(3)
            else:
                db_nominal_val = db_rsd_value

            # 2. So sánh Nominal bằng số học
            if db_nominal_val and parsed['nominal']:
                try:
                    db_nom_float = float(db_nominal_val)
                    ocr_nom_float = float(parsed['nominal'])
                    
                    if abs(db_nom_float - ocr_nom_float) < 0.001:
                        # 3. So sánh Dung sai bằng số học
                        # Chuyển về float, mặc định 0.0 nếu không có
                        ocr_up = float(parsed['upper']) if parsed['upper'] else 0.0
                        ocr_lo = float(parsed['lower']) if parsed['lower'] else 0.0
                        
                        if db_upper_val and db_lower_val:
                            db_up = float(db_upper_val)
                            db_lo = float(db_lower_val)
                            
                            # Cả hai dung sai phải khớp
                            if abs(db_up - ocr_up) < 0.001 and abs(db_lo - ocr_lo) < 0.001:
                                is_match = True
                            else:
                                is_match = False
                        else:
                            # DB không có dung sai, nếu OCR cũng xấp xỉ 0 thì khớp
                            if abs(ocr_up) < 0.001 and abs(ocr_lo) < 0.001:
                                is_match = True
                except:
                    is_match = False

            excel_data_compat = {
                "spec": db_nominal_val,
                "new_tolerance": db_rsd_value
            }

            if is_match:
                return 'success', 'Matched with Database', excel_data_compat
            else:
                return 'mismatch', f'Mismatch. DB: {db_rsd_value}', excel_data_compat

        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return 'error', str(e), None
        finally:
            conn.close()

if __name__ == "__main__":
    RSDValidator()
