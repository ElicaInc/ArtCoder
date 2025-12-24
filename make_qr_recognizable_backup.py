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

def create_code_target_m(message, content_image_path, output_path, image_size=DEFAULT_IMAGE_SIZE):
    """
    Create initial code_target_M.jpg from message and content image
    Enhanced to preserve center regions and maintain QR structure in right-side data areas
    """
    
    # Calculate dynamic module size
    module_size = image_size // 37
    
    # Load content image
    content_img = cv2.imread(content_image_path)
    if content_img is None:
        raise ValueError(f"Could not load content image: {content_image_path}")
    
    print(f"📷 Creating Enhanced Code Target M from:")
    print(f"   Message: '{message}'")
    print(f"   Content image: {content_image_path}")
    print(f"   Target size: {image_size}x{image_size} ({module_size}x{module_size} per module)")
    
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
    
    # Resize content image to match QR size
    content_resized = cv2.resize(content_img, (image_size, image_size))
    content_gray = cv2.cvtColor(content_resized, cv2.COLOR_BGR2GRAY)
    
    # Create content-influenced QR code
    result = np.zeros((image_size, image_size), dtype=np.uint8)
    
    print("🎨 Applying enhanced content influence with center preservation...")
    
    # Create enhanced function pattern masks
    function_mask = np.zeros((37, 37), dtype=bool)
    center_preserve_mask = np.zeros((37, 37), dtype=bool)
    data_priority_mask = np.zeros((37, 37), dtype=bool)
    
    # Finder patterns (3 corners) - 7x7 each with 1-module separators
    finder_positions = [(0, 0), (0, 30), (30, 0)]
    for pos_row, pos_col in finder_positions:
        if pos_row == 0 and pos_col == 30:  # Top-right
            function_mask[pos_row:pos_row+8, max(0, pos_col-1):] = True
        elif pos_row == 30 and pos_col == 0:  # Bottom-left
            function_mask[max(0, pos_row-1):, pos_col:pos_col+8] = True
        else:  # Top-left
            function_mask[pos_row:pos_row+8, pos_col:pos_col+8] = True
    
    # Timing patterns (row 6, col 6)
    function_mask[6, :] = True
    function_mask[:, 6] = True
    
    # Format information (row 8, col 8)
    function_mask[8, :] = True
    function_mask[:, 8] = True
    
    # Alignment pattern (center at 30,30 for version 5)
    align_center = 30
    align_size = 2  # 5x5 pattern, so ±2 from center
    function_mask[align_center-align_size:align_center+align_size+1, 
                  align_center-align_size:align_center+align_size+1] = True
    
    # Define enhanced center preservation area (middle 50% for better content preservation)
    center_start = int(37 * 0.25)
    center_end = int(37 * 0.75)
    center_preserve_mask[center_start:center_end, center_start:center_end] = True
    
    # Define right-side data priority areas (right 60% excluding function patterns)
    data_priority_mask[:, int(37 * 0.4):] = True
    data_priority_mask = data_priority_mask & ~function_mask  # Exclude function patterns
    
    # Add strong QR priority mask for rightmost regions (right 65%)
    strong_data_mask = np.zeros((37, 37), dtype=bool)
    strong_data_mask[:, int(37 * 0.35):] = True
    strong_data_mask = strong_data_mask & ~function_mask
    
    # Generate enhanced content-influenced pattern
    for i in range(37):
        for j in range(37):
            module_start_row = i * module_size
            module_end_row = module_start_row + module_size
            module_start_col = j * module_size  
            module_end_col = module_start_col + module_size
            
            if function_mask[i, j]:
                # Function pattern areas - strict QR pattern enforcement
                qr_value = 0 if qr_matrix[i, j] else 255
                result[module_start_row:module_end_row, module_start_col:module_end_col] = qr_value
                
            elif center_preserve_mask[i, j]:
                # Center preservation area - minimal QR modification, maximum content preservation
                content_patch = content_gray[module_start_row:module_end_row, module_start_col:module_end_col]
                qr_value = 0 if qr_matrix[i, j] else 255
                
                # Very subtle QR influence to preserve content appearance
                if qr_value == 0:  # Should be black
                    influenced = content_patch * 0.9  # Minimal darkening
                    influenced = np.clip(influenced, 30, 150)  # Allow more range
                else:  # Should be white
                    influenced = content_patch * 0.95 + 20  # Minimal lightening
                    influenced = np.clip(influenced, 120, 255)
                
                result[module_start_row:module_end_row, module_start_col:module_end_col] = influenced.astype(np.uint8)
                
            elif strong_data_mask[i, j]:\n                # Rightmost data areas (right 65%) - maximum QR structure preservation\n                qr_value = 0 if qr_matrix[i, j] else 255\n                content_patch = content_gray[module_start_row:module_end_row, module_start_col:module_end_col]\n                \n                # Very strong QR structure preservation\n                if qr_value == 0:  # Should be black module\n                    influenced = content_patch * 0.2 + qr_value * 0.6  # Very strong QR bias\n                    influenced = np.clip(influenced, 5, 80)  # Keep very dark\n                else:  # Should be white module\n                    influenced = content_patch * 0.3 + qr_value * 0.5  # Strong QR bias\n                    influenced = np.clip(influenced, 170, 255)  # Keep very light\n                \n                result[module_start_row:module_end_row, module_start_col:module_end_col] = influenced.astype(np.uint8)\n                \n            elif data_priority_mask[i, j]:
                # Right-side data priority areas - stronger QR maintenance
                qr_value = 0 if qr_matrix[i, j] else 255
                content_patch = content_gray[module_start_row:module_end_row, module_start_col:module_end_col]
                
                # Strong QR structure preservation with moderate content influence
                if qr_value == 0:  # Should be black module
                    influenced = content_patch * 0.4 + qr_value * 0.4  # Strong QR bias
                    influenced = np.clip(influenced, 10, 100)  # Keep dark
                else:  # Should be white module
                    influenced = content_patch * 0.5 + qr_value * 0.3  # Moderate content, strong QR
                    influenced = np.clip(influenced, 150, 255)  # Keep light
                
                result[module_start_row:module_end_row, module_start_col:module_end_col] = influenced.astype(np.uint8)
                
            else:
                # Regular data areas - balanced QR structure with content features
                qr_value = 0 if qr_matrix[i, j] else 255
                content_patch = content_gray[module_start_row:module_end_row, module_start_col:module_end_col]
                
                # Balanced approach for other areas
                if qr_value == 0:  # Should be black module
                    influenced = content_patch * 0.6  # Make darker
                    influenced = np.clip(influenced, 20, 120)  # Keep in dark range
                else:  # Should be white module
                    influenced = content_patch * 0.8 + 50  # Make lighter
                    influenced = np.clip(influenced, 130, 255)  # Keep in light range
                
                result[module_start_row:module_end_row, module_start_col:module_end_col] = influenced.astype(np.uint8)
    
    # Save the result
    cv2.imwrite(output_path, result)
    
    print(f"✅ Enhanced Code Target M created: {output_path}")
    print(f"📐 Size: {result.shape} ({module_size}px modules)")
    print(f"🎯 Center preservation applied to middle 40% region")
    print(f"🔍 Data region QR maintenance on right 60% area")
    print(f"🔧 Function patterns preserved for QR structure")
    print(f"📊 Processing summary:")
    print(f"   - Function patterns: {np.sum(function_mask)} modules")
    print(f"   - Center preserve: {np.sum(center_preserve_mask)} modules")
    print(f"   - Data priority: {np.sum(data_priority_mask)} modules")
    
    return result

def make_qr_recognizable(original_code_target_path, message, output_path, image_size=None):
    """
    Enhanced QR recognition fixer - preserves center while ensuring right-side data region QR integrity
    while preserving as much content appearance as possible
    """
    
    # Load the existing code target
    existing_img = cv2.imread(original_code_target_path, cv2.IMREAD_GRAYSCALE)
    if existing_img is None:
        raise ValueError(f"Could not load image: {original_code_target_path}")
    
    # Determine image size and module size
    if image_size is None:
        image_size = existing_img.shape[0]  # Assume square image
    module_size = image_size // 37
    
    print(f"📷 Loading: {original_code_target_path}")
    print(f"📐 Image shape: {existing_img.shape}")
    print(f"🔧 Module size: {module_size}px")
    
    # Resize if necessary
    if existing_img.shape[0] != image_size or existing_img.shape[1] != image_size:
        existing_img = cv2.resize(existing_img, (image_size, image_size))
        print(f"🔄 Resized to: {image_size}x{image_size}")
    
    # Generate reference QR code
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
    qr_reference = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(37):
        for j in range(37):
            value = 0 if qr_matrix[i, j] else 255  # 0=black, 255=white
            start_row = i * module_size
            end_row = start_row + module_size
            start_col = j * module_size
            end_col = start_col + module_size
            qr_reference[start_row:end_row, start_col:end_col] = value
    
    # Start with existing image
    result = existing_img.copy()
    
    print("🔧 Applying aggressive QR recognition fixes...")
    
    # Step 1: FORCE restore all critical patterns
    
    # 1.1 Finder Patterns - Complete restoration with proper separators
    finder_positions = [(0, 0), (0, 30), (30, 0)]
    
    for pos_row, pos_col in finder_positions:
        # 7x7 finder pattern + 1-module separator = 8x8 area
        pixel_row = pos_row * module_size
        pixel_col = pos_col * module_size
        
        if pos_row == 0 and pos_col == 30:  # Top-right
            # Include left separator
            result[pixel_row:pixel_row + 8*module_size, pixel_col-module_size:] = \
                qr_reference[pixel_row:pixel_row + 8*module_size, pixel_col-module_size:]
        elif pos_row == 30 and pos_col == 0:  # Bottom-left  
            # Include top separator
            result[pixel_row-module_size:, pixel_col:pixel_col + 8*module_size] = \
                qr_reference[pixel_row-module_size:, pixel_col:pixel_col + 8*module_size]
        else:  # Top-left
            # Include right and bottom separators
            result[pixel_row:pixel_row + 8*module_size, pixel_col:pixel_col + 8*module_size] = \
                qr_reference[pixel_row:pixel_row + 8*module_size, pixel_col:pixel_col + 8*module_size]
        
        print(f"   ✅ Restored finder + separator at ({pos_row}, {pos_col})")
    
    # 1.2 Timing Patterns - Full restoration
    timing_pixel = 6 * module_size
    result[timing_pixel:timing_pixel + module_size, :] = \
        qr_reference[timing_pixel:timing_pixel + module_size, :]
    result[:, timing_pixel:timing_pixel + module_size] = \
        qr_reference[:, timing_pixel:timing_pixel + module_size]
    print("   ✅ Restored timing patterns")
    
    # 1.3 Format Information - Complete restoration
    # These are absolutely critical for QR recognition
    
    # Horizontal format strips
    result[8*module_size:9*module_size, :] = \
        qr_reference[8*module_size:9*module_size, :]
    # Vertical format strips  
    result[:, 8*module_size:9*module_size] = \
        qr_reference[:, 8*module_size:9*module_size]
    
    print("   ✅ Restored format information")
    
    # 1.4 Alignment Pattern
    align_center = 30
    align_pixel_center = align_center * module_size
    align_size = 5 * module_size
    start_row = align_pixel_center - align_size // 2
    end_row = start_row + align_size  
    start_col = align_pixel_center - align_size // 2
    end_col = start_col + align_size
    
    result[start_row:end_row, start_col:end_col] = \
        qr_reference[start_row:end_row, start_col:end_col]
    print("   ✅ Restored alignment pattern")
    
    # Step 2: Aggressive data region correction
    # Apply strong binarization to data areas while preserving content patterns
    
    print("🎯 Applying smart data region enhancement...")
    
    # Create comprehensive exclusion mask for protected areas
    protected_mask = np.zeros((image_size, image_size), dtype=bool)
    
    # Protect finder areas (8x8 including separators)
    for pos_row, pos_col in finder_positions:
        pixel_row = pos_row * module_size
        pixel_col = pos_col * module_size
        if pos_row == 0 and pos_col == 30:  # Top-right
            protected_mask[pixel_row:pixel_row + 8*module_size, pixel_col-module_size:] = True
        elif pos_row == 30 and pos_col == 0:  # Bottom-left
            protected_mask[pixel_row-module_size:, pixel_col:pixel_col + 8*module_size] = True
        else:  # Top-left
            protected_mask[pixel_row:pixel_row + 8*module_size, pixel_col:pixel_col + 8*module_size] = True
    
    # Protect timing patterns
    protected_mask[6*module_size:7*module_size, :] = True
    protected_mask[:, 6*module_size:7*module_size] = True
    
    # Protect format information
    protected_mask[8*module_size:9*module_size, :] = True
    protected_mask[:, 8*module_size:9*module_size] = True
    
    # Protect alignment pattern
    protected_mask[start_row:end_row, start_col:end_col] = True
    
    # Get data regions (inverse of protected)
    data_mask = ~protected_mask
    
    # Apply module-based binarization to data regions
    print("   📊 Applying module-based binarization...")
    
    for i in range(37):
        for j in range(37):
            # Skip protected modules
            module_start_row = i * module_size
            module_end_row = module_start_row + module_size
            module_start_col = j * module_size
            module_end_col = module_start_col + module_size
            
            # Check if this module is in data region
            module_center = (module_start_row + module_size//2, module_start_col + module_size//2)
            if not data_mask[module_center]:
                continue
            
            # Get current module data
            module_data = result[module_start_row:module_end_row, module_start_col:module_end_col]
            module_mean = np.mean(module_data)
            
            # Get reference value for this module
            reference_value = qr_reference[module_start_row, module_start_col]  # 0 or 255
            
            # Smart binarization: blend content features with QR requirements
            if reference_value == 0:  # Should be black
                # Make it darker, but preserve some content variation
                if module_mean > 128:  # If it's currently light
                    # Force towards black but keep some content texture
                    enhanced = module_data * 0.3  # Make much darker
                    enhanced = np.clip(enhanced, 0, 100)  # Cap at dark gray
                else:
                    # Already dark, just enhance contrast
                    enhanced = module_data * 0.7
                    enhanced = np.clip(enhanced, 0, 80)
            else:  # Should be white
                # Make it lighter, preserve content variation
                if module_mean < 128:  # If it's currently dark
                    # Force towards white but keep some content texture
                    enhanced = 255 - (255 - module_data) * 0.3  # Make much lighter
                    enhanced = np.clip(enhanced, 155, 255)  # Cap at light gray
                else:
                    # Already light, enhance contrast
                    enhanced = 255 - (255 - module_data) * 0.7
                    enhanced = np.clip(enhanced, 175, 255)
            
            result[module_start_row:module_end_row, module_start_col:module_end_col] = enhanced.astype(np.uint8)
    
    print("   ✅ Applied module-based enhancement")
    
    # Step 3: Add quiet zone border (important for recognition)
    print("🔲 Adding quiet zone border...")
    
    # Create bordered image with quiet zone
    border_size = module_size * 2  # 2 modules worth of quiet zone
    bordered_size = image_size + 2 * border_size
    bordered_result = np.full((bordered_size, bordered_size), 255, dtype=np.uint8)  # White border
    
    # Place QR code in center
    bordered_result[border_size:border_size + image_size, border_size:border_size + image_size] = result
    
    print(f"   ✅ Added {border_size}px quiet zone border")
    
    # Save both versions
    cv2.imwrite(output_path, result)  # Original size
    bordered_output_path = output_path.replace('.jpg', '_bordered.jpg')
    cv2.imwrite(bordered_output_path, bordered_result)  # With border
    
    print(f"\n🎉 Enhanced QR codes saved:")
    print(f"   📄 Standard: {output_path}")
    print(f"   🔲 With border: {bordered_output_path}")
    
    print(f"\n🔍 Applied enhancements:")
    print(f"   ✅ Complete finder pattern restoration")
    print(f"   ✅ Complete timing pattern restoration") 
    print(f"   ✅ Complete format information restoration")
    print(f"   ✅ Complete alignment pattern restoration")
    print(f"   ✅ Smart module-based data region enhancement")
    print(f"   ✅ Added quiet zone border for better recognition")
    
    print(f"\n📱 Both versions should be fully scannable!")
    print(f"💡 Try the bordered version first: {bordered_output_path}")
    
    return result, bordered_result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        # Create new code_target_M.jpg
        print("🚀 Creating new Code Target M...")
        message = DEFAULT_MESSAGE
        content_path = DEFAULT_CONTENT_PATH
        output_path = DEFAULT_OUTPUT_PATH
        
        try:
            result = create_code_target_m(message, content_path, output_path, image_size=DEFAULT_IMAGE_SIZE)
            print(f"\n✅ Successfully created {output_path}")
            print(f"💡 To enhance readability, run: python make_qr_recognizable.py fix")
        except Exception as e:
            print(f"❌ Error creating code target: {e}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "fix":
        # Fix existing code_target_M.jpg to make it recognizable
        print("🔧 Fixing existing Code Target M for recognition...")
        original_path = DEFAULT_INITIAL_PATH
        message = DEFAULT_MESSAGE
        output_path = DEFAULT_OUTPUT_PATH
        
        try:
            standard_img, bordered_img = make_qr_recognizable(original_path, message, output_path, image_size=DEFAULT_IMAGE_SIZE)
            print(f"\n✅ Successfully enhanced {original_path}")
            print(f"🎯 Standard version: {output_path}")
            print(f"🔲 Bordered version: {output_path.replace('.jpg', '_bordered.jpg')}")
            print(f"\n💡 For ArtCoder, use: --code_img_path {output_path}")
        except Exception as e:
            print(f"❌ Error fixing code target: {e}")
    
    else:
        # Default behavior - create and fix
        print("🚀 Creating and fixing Code Target M...")
        message = DEFAULT_MESSAGE
        content_path = DEFAULT_CONTENT_PATH
        initial_output = DEFAULT_INITIAL_PATH
        final_output = DEFAULT_OUTPUT_PATH
        
        try:
            # Step 1: Create initial code target
            print("\n📝 Step 1: Creating initial Code Target M...")
            create_code_target_m(message, content_path, initial_output, image_size=DEFAULT_IMAGE_SIZE)
            
            # Step 2: Fix for recognition
            print("\n🔧 Step 2: Enhancing for QR recognition...")
            standard_img, bordered_img = make_qr_recognizable(initial_output, message, final_output, image_size=DEFAULT_IMAGE_SIZE)
            
            print(f"\n🎉 Complete workflow finished!")
            print(f"📄 Initial version: {initial_output}")
            print(f"🎯 Enhanced version: {final_output}")
            print(f"🔲 Bordered version: {final_output.replace('.jpg', '_bordered.jpg')}")
            print(f"\n💡 For ArtCoder, use: --code_img_path {final_output}")
            
        except Exception as e:
            print(f"❌ Error in workflow: {e}")
    
    # Display current settings
    print(f"\n📋 Current settings:")
    print(f"   🔗 Message: {DEFAULT_MESSAGE}")
    print(f"   🖼️  Content: {DEFAULT_CONTENT_PATH}")
    print(f"   📄 Output: {DEFAULT_OUTPUT_PATH}")
    print(f"   📝 Initial: {DEFAULT_INITIAL_PATH}")
    print(f"\n💡 To change message, edit DEFAULT_MESSAGE at the top of this file")
            
    print(f"\n📚 Usage options:")
    print(f"   python make_qr_recognizable.py        # Create and fix (default)")
    print(f"   python make_qr_recognizable.py create # Create only")
    print(f"   python make_qr_recognizable.py fix    # Fix existing")