import argparse
import utils as utils
from Artcoder import artcoder
import qrcode
import os
from PIL import Image
import cv2
import numpy as np
from qrcode.constants import ERROR_CORRECT_H

"""
 基本的な使用（自動QR生成）
python main.py --auto_qr

# カスタムメッセージでQR生成
python main.py --auto_qr --qr_message "https://example.com"

# 完全カスタマイズ
python main.py --auto_qr --qr_message "Your URL" --style_img_path ./style/mosaic.jpg --epoch 100
python main.py --auto_qr --style_img_path ./style/s27.jpg --epoch 10000
python main.py --auto_qr \
    --style_img_path ./style/oden16.jpg \
    --epoch 20000 \
    --code_weight 1e12 \
    --content_weight 1e8 \
    --style_weight 1e14 \



# 重みパラメータの範囲と効果
style_weight (デフォルト: 1e15)
範囲: 通常 1e12 ～ 1e18
効果:
高い値: スタイル画像の質感・色調・パターンが強く反映される
低い値: スタイルの影響が弱くなり、元の画像に近くなる
content_weight (デフォルト: 1e8)
範囲: 通常 1e6 ～ 1e10
効果:
高い値: コンテンツ画像の構造・形状が強く保持される
低い値: コンテンツの構造が曖昧になり、スタイルが優勢になる
code_weight (デフォルト: 1e12)
範囲: 通常 1e10 ～ 1e15
効果:
高い値: QRコードの可読性が強く保たれる（重要）
低い値: QRコードが読み取れなくなるリスクが高まる

"""

def create_pure_qr_code(message, output_path, image_size=592):
    """
    Create pure QR code without background content
    """
    # Calculate dynamic module size for exact fit
    module_size = image_size / 37  # Use float division for exact fit
    
    print(f"📷 Creating Pure QR Code:")
    print(f"   Message: '{message}'")
    print(f"   Target size: {image_size}x{image_size} ({module_size:.2f}x{module_size:.2f} per module)")
    print(f"   Output: {output_path}")
    
    # Generate basic QR code
    qr = qrcode.QRCode(
        version=5,
        error_correction=ERROR_CORRECT_H,
        box_size=module_size,
        border=0,
    )
    qr.add_data(message)
    qr.make(fit=True)
    
    # Get QR matrix and convert to image array
    qr_matrix = np.array(qr.get_matrix(), dtype=int)
    
    # Create pure QR code without content influence
    result = np.zeros((image_size, image_size), dtype=np.uint8)
    
    print("🎨 Creating pure QR structure...")
    
    # Generate pure QR pattern without content influence
    for i in range(37):
        for j in range(37):
            module_start_row = int(i * module_size)
            module_end_row = int((i + 1) * module_size)
            module_start_col = int(j * module_size)
            module_end_col = int((j + 1) * module_size)
            
            # Pure QR pattern - black (0) or white (255)
            qr_value = 0 if qr_matrix[i, j] else 255
            result[module_start_row:module_end_row, module_start_col:module_end_col] = qr_value
    
    # Convert to 3-channel RGB for VGG compatibility
    result_rgb = np.stack([result, result, result], axis=2)
    
    # Save as RGB image
    cv2.imwrite(output_path, result_rgb)
    
    print(f"✅ Pure QR Code created: {output_path}")
    print(f"📐 Size: {result.shape} ({module_size:.2f}px modules)")
    
    return output_path

def generate_qr_code_v5(text, output_path, module_size=16):
    """
    Generate QR code version 5 (37x37 modules) from text
    """
    # QR code version 5 settings
    qr = qrcode.QRCode(
        version=5,  # QR code version 5 (37x37 modules)
        error_correction=qrcode.constants.ERROR_CORRECT_M,  # Medium error correction
        box_size=module_size,  # Size of each module in pixels
        border=0,  # Border size (0 for exact 37x37 modules)
    )
    
    qr.add_data(text)
    qr.make(fit=True)
    
    # Create QR code image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to RGB mode to ensure 3 dimensions
    qr_img = qr_img.convert('RGB')
    
    # Ensure the image size is exactly module_size * 37
    target_size = module_size * 37
    qr_img = qr_img.resize((target_size, target_size), Image.LANCZOS)
    
    # Save QR code
    qr_img.save(output_path)
    print(f"QR code version 5 generated and saved to: {output_path}")
    print(f"QR code size: {qr_img.size[0]}x{qr_img.size[1]} pixels ({37}x{37} modules)")
    print(f"QR code mode: {qr_img.mode}")
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qr_text', help="text to encode in QR code (if provided, generates QR from text)", type=str, default=None)
    parser.add_argument('--qr_message', help="message to encode in pure QR code (alternative to qr_text)", type=str, default="https://oden.pref.oita.jp/")
    parser.add_argument('--qr_output_path', help="path to save generated QR code (default: './generated_qr.png')", type=str, default='./generated_qr.png')
    parser.add_argument('--auto_qr', help="automatically generate pure QR code if no code_img_path exists", action='store_true')
    parser.add_argument('--style_img_path', help="path to input style target (default: './images/style.jpg')", type=str,
                        default='./images/style.jpg')
    parser.add_argument('--content_img_path', help="path to input content target (default: './images/content.jpg')", type=str,
                        default='./images/content.jpg')
    parser.add_argument('--code_img_path', help="path to input code target (default: './images/code.jpg')", type=str,
                        default='./images/code.jpg')
    parser.add_argument('--output_dir', help='path to save output stylized QR code', type=str,
                        default='./output/')
    parser.add_argument('--learning_rate',
                        help='learning rate (default: 0.01)',
                        type=int, default=0.01)
    parser.add_argument('--style_weight', help='style_weight', type=int, default=1e15)
    parser.add_argument('--content_weight', help='content_weight', type=int, default=1e8)
    parser.add_argument('--code_weight', help='code_weight', type=int, default=1e12)

    parser.add_argument('--module_size',
                        help='the resolution of each square module of a QR code (default: 16)',
                        type=int, default=16)
    parser.add_argument('--module_number',
                        help='Number of QR code modules per side (default: 37)',
                        type=int, default=37)
    parser.add_argument('--epoch', help='epoch number (default: 1000)', type=int,
                        default=1000)
    parser.add_argument('--discriminate_b',
                        help="for black modules, pixels' gray values under discriminate_b will be discriminated to error modules to activate sub-code-losses (discriminate_b in [0-128])",
                        type=int,
                        default=70)
    parser.add_argument('--discriminate_w',
                        help="for white modules, pixels' gray values over discriminate_w will be discriminated to error modules to activate sub-code-losses (discriminate_w in [128-255])",
                        type=int,
                        default=180)
    parser.add_argument('--correct_b',
                        help="for black module, correct error modules' gray value to correct_b (correct_b < discriminate_b)",
                        type=int,
                        default=40)
    parser.add_argument('--correct_w',
                        help="for white module, correct error modules' gray value to correct_w (correct_w > discriminate_w)",
                        type=int,
                        default=220)
    parser.add_argument('--use_activation_mechanism',
                        help="whether to use the activation mechanism (1 means use and other numbers mean not)",
                        type=int,
                        default=1)

    args = parser.parse_args()
    utils.print_options(opt=args)

    # QR code generation logic
    code_path = args.code_img_path
    
    # Priority: qr_text > auto_qr > existing code_img_path
    if args.qr_text:
        print("=== QR Code Generation (Legacy Mode) ===")
        print(f"Generating QR code version 5 from text: '{args.qr_text}'")
        code_path = generate_qr_code_v5(
            text=args.qr_text,
            output_path=args.qr_output_path,
            module_size=args.module_size
        )
        print("=== Starting Stylized QR Generation ===")
    
    elif args.auto_qr or not os.path.exists(args.code_img_path):
        print("=== Pure QR Code Generation ===")
        # Calculate image size from module size
        target_size = args.module_size * 37
        if args.auto_qr:
            print(f"Auto-generating pure QR code from message: '{args.qr_message}'")
        else:
            print(f"Code image not found at '{args.code_img_path}', generating pure QR code")
            print(f"Using default message: '{args.qr_message}'")
        
        code_path = create_pure_qr_code(
            message=args.qr_message,
            output_path=args.code_img_path,  # Save to original path
            image_size=target_size
        )
        print("=== Starting Stylized QR Generation ===")
    
    else:
        print(f"=== Using Existing Code Image ===")
        print(f"Code image: {args.code_img_path}")

    # Auto-detect module_size from code image if it exists
    actual_module_size = args.module_size
    if code_path and code_path.endswith(('.jpg', '.jpeg', '.png')):
        try:
            from PIL import Image
            code_img = Image.open(code_path)
            detected_module_size = code_img.size[0] // 37
            if detected_module_size != args.module_size:
                print(f"Auto-detected module_size: {detected_module_size} (overriding default {args.module_size})")
                actual_module_size = detected_module_size
        except Exception as e:
            print(f"Could not auto-detect module size: {e}")

    artcoder(STYLE_IMG_PATH=args.style_img_path, CONTENT_IMG_PATH=args.content_img_path, CODE_PATH=code_path,
             OUTPUT_DIR=args.output_dir, LEARNING_RATE=args.learning_rate, CONTENT_WEIGHT=args.content_weight,
             STYLE_WEIGHT=args.style_weight, CODE_WEIGHT=args.code_weight, MODULE_SIZE=actual_module_size,
             MODULE_NUM=args.module_number, EPOCHS=args.epoch, Dis_b=args.discriminate_b, Dis_w=args.discriminate_w,
             Correct_b=args.correct_b, Correct_w=args.correct_w, USE_ACTIVATION_MECHANISM=args.use_activation_mechanism)
