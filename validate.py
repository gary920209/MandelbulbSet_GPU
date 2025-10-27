import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: validate.py <test.png> <valid.png> [threshold=0.97]")
        sys.exit(2)

    test_path  = sys.argv[1]
    valid_path = sys.argv[2]
    try:
        threshold = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.97
    except ValueError:
        print("Invalid threshold. Must be a float between 0 and 1.")
        sys.exit(2)

    # Lazy imports to allow script to run even if Pillow/NumPy aren’t present yet.
    try:
        from PIL import Image
    except Exception as e:
        print("Pillow (PIL) is required. Try:  python3 -m pip install --user pillow")
        print(f"Import error: {e}")
        sys.exit(2)

    # Load and normalize to RGBA
    try:
        im_test  = Image.open(test_path).convert("RGBA")
        im_valid = Image.open(valid_path).convert("RGBA")
    except Exception as e:
        print(f"Failed to open images: {e}")
        sys.exit(2)

    if im_test.size != im_valid.size:
        print(f"DIFF SIZE: {im_test.size} vs {im_valid.size}")
        sys.exit(2)

    width, height = im_test.size
    n_pix = width * height

    # Fast path with NumPy if available
    try:
        import numpy as np  # optional
        a = np.asarray(im_test, dtype=np.uint8)   # H x W x 4
        b = np.asarray(im_valid, dtype=np.uint8)  # H x W x 4
        eq = np.all(a == b, axis=2)               # H x W bool (per-pixel RGBA equality)
        match = int(eq.sum())
    except Exception:
        # Fallback: pure-Python byte comparison
        b1 = im_test.tobytes()
        b2 = im_valid.tobytes()
        match = 0
        # Compare per pixel (4 bytes per pixel)
        for i in range(0, len(b1), 4):
            if b1[i:i+4] == b2[i:i+4]:
                match += 1

    ratio = match / n_pix
    print(f"Equal pixels: {match}/{n_pix} ({ratio*100:.2f}%)  threshold: {threshold*100:.2f}%")
    sys.exit(0 if ratio >= threshold else 1)

if __name__ == "__main__":
    main()