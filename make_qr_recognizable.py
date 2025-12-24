import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_H
from PIL import Image

# ===== 設定 =====
DEFAULT_MESSAGE = "https://oden.pref.oita.jp/"
DEFAULT_CONTENT_PATH = "./images/content.jpg"
DEFAULT_OUTPUT_PATH = "./images/code.jpg"
DEFAULT_INITIAL_PATH = "./images/code_target_M.jpg"
DEFAULT_IMAGE_SIZE = 512
# ===============

def create_code_target_m(message, output_path, image_size=DEFAULT_IMAGE_SIZE):
    """
    Create initial code_target_M.jpg from message only (no background content)
    Pure QR code with enhanced structure for better recognition
    """
    
    # Calculate dynamic module size for exact fit
    module_size = image_size / 37  # Use float division for exact fit
    
    print(f"📷 Creating Pure QR Code Target M:")
    print(f"   Message: '{message}'")
    print(f"   Target size: {image_size}x{image_size} ({module_size:.2f}x{module_size:.2f} per module)")
    print(f"   No background content - pure QR structure")
    
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
    
    # Save the result as RGB image (keeping grayscale values for color compatibility)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_path, result_rgb)
    
    print(f"✅ Pure QR Code Target M created: {output_path}")
    print(f"📐 Size: {result.shape} ({module_size:.2f}px modules)")
    print(f"🔍 Pure QR structure - no background content")
    print(f"🎨 Grayscale output with color compatibility")
    print(f"📊 QR modules: {37 * 37} total")
    
    return result

if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create Enhanced Code Target M')
    parser.add_argument('command', nargs='?', default='create', help='Command to run (create)')
    parser.add_argument('--target_size', type=int, default=DEFAULT_IMAGE_SIZE, 
                       help=f'Target image size (default: {DEFAULT_IMAGE_SIZE})')
    parser.add_argument('--message', type=str, default=DEFAULT_MESSAGE,
                       help=f'QR code message (default: {DEFAULT_MESSAGE})')
    parser.add_argument('--content_path', type=str, default=DEFAULT_CONTENT_PATH,
                       help=f'Content image path (default: {DEFAULT_CONTENT_PATH})')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH,
                       help=f'Output path (default: {DEFAULT_OUTPUT_PATH})')
    
    args = parser.parse_args()
    
    if args.command == "create":
        # Create new code_target_M.jpg
        print("🚀 Creating new Enhanced Code Target M...")
        message = args.message
        content_path = args.content_path
        output_path = args.output_path
        target_size = args.target_size
        
        try:
            result = create_code_target_m(message, output_path, image_size=target_size)
            print(f"\n✅ Successfully created {output_path}")
            print(f"🎯 Pure QR code without background content")
            print(f"📐 Target size: {target_size}x{target_size}")
            module_size = target_size / 37
            print(f"📏 Module size: {module_size:.2f}px per QR module")
        except Exception as e:
            print(f"❌ Error creating code target: {e}")
    
    else:
        # Default behavior - just create
        print("🚀 Creating Pure QR Code Target M...")
        message = args.message
        output_path = args.output_path
        target_size = args.target_size
        
        try:
            create_code_target_m(message, output_path, image_size=target_size)
            print(f"\n🎉 Pure QR Code Target M creation finished!")
            print(f"📄 Created: {output_path}")
            print(f"📐 Size: {target_size}x{target_size}")
            print(f"🔍 Pure QR structure - no background")
            print(f"\n💡 For ArtCoder, use: --code_img_path {output_path}")
            
        except Exception as e:
            print(f"❌ Error in workflow: {e}")
    
    # Display current settings
    print(f"\n📋 Current settings:")
    print(f"   🔗 Message: {message}")
    print(f"   📄 Output: {output_path}")
    print(f"   📐 Image size: {target_size}x{target_size}")