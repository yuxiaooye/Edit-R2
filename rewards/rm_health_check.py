import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_grpo.edival_client import check_server_health, evaluate_via_server, evaluate_consistency_via_server
from PIL import Image, ImageDraw

DEFAULT_SERVER_URL = "http://127.0.0.1:12342"
SERVER_URL = os.getenv("EDIVAL_SERVER_URL", DEFAULT_SERVER_URL)
DEFAULT_GROUNDINGDINO_URL = "http://127.0.0.1:12343"
GROUNDINGDINO_URL = os.getenv("GROUNDINGDINO_URL", DEFAULT_GROUNDINGDINO_URL)
print('>>>SERVER_URL:', SERVER_URL)

# Create the output directory
output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("EDiVal Test (with image saving)")
print("="*60)
print(f"Images will be saved to: {output_dir}")

# Check server health
print("\n[1/3] Checking main server health status...")
health = check_server_health(server_url=SERVER_URL)
print(f"  Status: {health}")

if health.get("status") != "healthy":
    print("\n  ⚠ Service not ready, please make sure:")
    print("    1. bash start_servers.sh (start the sub-services)")
    print("    2. python reward_server/edival_reward_server.py (start the main service)")
    sys.exit(1)

print("  ✓ Service is healthy")

# Create test images
print("\n[2/3] Creating test images...")

def draw_tree(draw, x_center, y_trunk_bottom=450, trunk_width=32, trunk_height=100, crown_height=150):
    """Draw a tree"""
    # Trunk (brown rectangle)
    trunk_left = x_center - trunk_width // 2
    trunk_right = x_center + trunk_width // 2
    trunk_top = y_trunk_bottom - trunk_height
    draw.rectangle([trunk_left, trunk_top, trunk_right, y_trunk_bottom], 
                   fill='brown', outline='black', width=2)
    
    # Crown (green triangle)
    crown_bottom = trunk_top
    crown_top = crown_bottom - crown_height
    left_point = (x_center - trunk_width * 3, crown_bottom)
    right_point = (x_center + trunk_width * 3, crown_bottom)
    top_point = (x_center, crown_top)
    draw.polygon([left_point, right_point, top_point], 
                 fill='green', outline='darkgreen', width=2)

def draw_dog(draw, x_body_center, y_body_bottom=400, body_width=100, body_height=50):
    """Draw a detailed dog (including ears, tail, eyes, nose)"""
    # Dog body (ellipse)
    body_left = x_body_center - body_width // 2
    body_right = x_body_center + body_width // 2
    body_top = y_body_bottom - body_height
    draw.ellipse([body_left, body_top, body_right, y_body_bottom], 
                 fill='sandybrown', outline='black', width=2)
    
    # Dog head (ellipse) - located at the upper-left of the body
    head_size = body_height * 1.2
    head_left = body_left - head_size * 0.3
    head_right = head_left + head_size
    head_top = body_top - head_size * 0.8
    head_bottom = head_top + head_size
    draw.ellipse([head_left, head_top, head_right, head_bottom], 
                 fill='sandybrown', outline='black', width=2)
    
    # Dog ears (two small ellipses)
    ear_width = head_size * 0.25
    ear_height = head_size * 0.35
    # Left ear
    left_ear_x = head_left + head_size * 0.15
    draw.ellipse([left_ear_x, head_top - ear_height * 0.3, 
                  left_ear_x + ear_width, head_top + ear_height * 0.7], 
                 fill='sandybrown', outline='black', width=2)
    # Right ear
    right_ear_x = head_left + head_size * 0.6
    draw.ellipse([right_ear_x, head_top - ear_height * 0.3, 
                  right_ear_x + ear_width, head_top + ear_height * 0.7], 
                 fill='sandybrown', outline='black', width=2)
    
    # Dog tail (ellipse) - located at the back-right of the body
    tail_x = body_right + 10
    tail_y = body_top + body_height * 0.3
    draw.ellipse([tail_x, tail_y, tail_x + 40, tail_y + 20], 
                 fill='sandybrown', outline='black', width=2)
    
    # Eyes (two small black ellipses)
    eye_size = head_size * 0.08
    left_eye_x = head_left + head_size * 0.25
    right_eye_x = head_left + head_size * 0.55
    eye_y = head_top + head_size * 0.35
    draw.ellipse([left_eye_x, eye_y, left_eye_x + eye_size, eye_y + eye_size], fill='black')
    draw.ellipse([right_eye_x, eye_y, right_eye_x + eye_size, eye_y + eye_size], fill='black')
    
    # Nose (small black ellipse)
    nose_size = head_size * 0.12
    nose_x = head_left + head_size * 0.4
    nose_y = head_top + head_size * 0.55
    draw.ellipse([nose_x, nose_y, nose_x + nose_size, nose_y + nose_size], fill='black')

# Reference image (tree) - original image
ref_img = Image.new('RGB', (512, 512), color='lightblue')
draw = ImageDraw.Draw(ref_img)
draw_tree(draw, 256)  # Tree in the middle
ref_path = os.path.join(output_dir, "01_reference_tree.png")
ref_img.save(ref_path)
print(f"  ✓ Reference image saved: {ref_path}")

# Generated image (Case A: expected success - dog on the left of the tree)
gen_success = Image.new('RGB', (512, 512), color='lightblue')
draw = ImageDraw.Draw(gen_success)
# Tree on the right
draw_tree(draw, 350)  
# Dog on the left (using detailed drawing)
draw_dog(draw, 150)
success_path = os.path.join(output_dir, "02_gen_with_dog_left.png")
gen_success.save(success_path)
print(f"  ✓ Expected-success case saved: {success_path}")


# Generated image (Case B: expected failure - no dog)
gen_fail = Image.new('RGB', (512, 512), color='lightblue')
draw = ImageDraw.Draw(gen_fail)
draw_tree(draw, 256)
fail_path = os.path.join(output_dir, "03_gen_no_dog.png")
gen_fail.save(fail_path)
print(f"  ✓ Expected-failure case saved: {fail_path}")

# Generated image (Case C: dog on the right - wrong position)
gen_wrong_pos = Image.new('RGB', (512, 512), color='lightblue')
draw = ImageDraw.Draw(gen_wrong_pos)
# Tree on the left
draw_tree(draw, 162)
# Dog on the right
draw_dog(draw, 380)
wrong_pos_path = os.path.join(output_dir, "04_gen_dog_right.png")
gen_wrong_pos.save(wrong_pos_path)
print(f"  ✓ Wrong-position case saved: {wrong_pos_path}")

# Call the evaluation
print("\n[3/3] Calling EDiVal evaluation...")

metadata = {
    "task_type": "subject_add",
    "formatted_instruction": "Add [dog] on the [left] of [tree]",
    "unchanged_objects": "tree",
    "all_objects": "tree, dog"
}

test_cases = [
    ("Expected success (dog on the left)", gen_success),
    #("Expected failure (no dog)", gen_fail),
    #("Expected failure (dog on the right)", gen_wrong_pos),
]

print("\n  Test instruction: Add dog on the left of the tree\n")

for case_name, test_img in test_cases:
    print(f"  Test: {case_name}")
    try:
        scores, reasons = evaluate_via_server(
            [ref_img], [test_img],
            ["Add dog on the left of the tree"],
            [metadata],
            server_url=SERVER_URL,
            timeout=120
        )

        object_results, background_results = evaluate_consistency_via_server(
            [ref_img], [test_img],
            [metadata],
            server_url=SERVER_URL,
            timeout=120
        )
        
        print(f"    EdiVal-IF score: {scores[0]}")
        print(f"    Reason: {reasons[0][:150]}...")
        print(f"    object_results:", object_results)
        print(f"    background_results:", background_results)
        
    except Exception as e:
        print(f"    Error: {e}")
    print()

# Extra: directly test GroundingDINO detection
import requests

def test_dino_detection(image, target_object, case_name):
    """Directly call the GroundingDINO service for testing"""
    try:
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_b64 = f"data:image/png;base64,{img_str}"
        
        response = requests.post(
            f"{GROUNDINGDINO_URL}/detect",
            json={
                "image": image_b64,
                "target_object": target_object,
                "threshold": 0.3,
                "return_all": True
            },
            timeout=10
        )
        
        result = response.json()
        if result.get("success"):
            detections = result.get("detections", {})
            num_detections = len(detections.get("label", []))
            scores = detections.get("score", [])
            boxes = detections.get("box", [])
            
            print(f"  {case_name}:")
            print(f"    Detected {num_detections} '{target_object}'")
            if scores:
                print(f"    Confidence: {[f'{s:.3f}' for s in scores]}")
            if boxes:
                print(f"    Bounding boxes: {boxes}")
                # Draw bounding boxes and save (for annotation contribution)
                vis_img = image.copy()
                vis_draw = ImageDraw.Draw(vis_img)
                w, h = vis_img.size
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
                    vis_draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    vis_draw.text((x1, y1 - 12), f"{target_object} {score:.3f}", fill='red')
                safe_name = case_name.replace(' ', '_').replace("'", '').replace('(', '').replace(')', '')
                vis_path = os.path.join(output_dir, f"dino_{safe_name}_{target_object}.png")
                vis_img.save(vis_path)
                print(f"    Visualization saved: {vis_path}")
        else:
            print(f"  {case_name}: Detection failed - {result.get('error')}")
            
    except Exception as e:
        print(f"  {case_name}: Request failed - {e}")
