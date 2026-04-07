import cv2
import numpy as np
import torch
import torch.nn.functional as F
import struct


img = cv2.imread("rose.jpg")
if img is None:
    raise FileNotFoundError("rose.jpg not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
inH, inW = gray.shape
outH, outW = inH * 2, inW * 2

print(f"Input  : {inH} x {inW}")
print(f"Output : {outH} x {outW}")


def load_bin(path):
    with open(path, "rb") as f:
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).copy().reshape(rows, cols)
    print(f"Loaded: {path}  [{rows} x {cols}]")
    return data

cpp_fwd  = load_bin("cpp_forward.bin")   
grad_out = load_bin("grad_output.bin")   
cpp_bwd  = load_bin("cpp_backward.bin")  

#PyTorch forward
t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()
t.requires_grad_(True)

pt_out = F.interpolate(t, size=(outH, outW), mode='bicubic',
                       align_corners=False, antialias=False)

# PyTorch backward 
grad_out_tensor = torch.from_numpy(grad_out).unsqueeze(0).unsqueeze(0)
pt_out.backward(grad_out_tensor)

pt_fwd = pt_out.detach().squeeze().numpy()   # (outH, outW)
pt_bwd = t.grad.detach().squeeze().numpy()   # (inH,  inW)


def validate(A, B, label):
    diff = np.abs(A.astype(np.float32) - B.astype(np.float32))
    print(f"\n[{label}]")
    print(f"  Mean Error : {diff.mean():.6f}")
    print(f"  Max  Error : {diff.max():.6f}")
    print(f"  Total Error: {diff.sum():.4f}")
    print(f"  A range    : [{A.min():.4f}, {A.max():.4f}]")
    print(f"  B range    : [{B.min():.4f}, {B.max():.4f}]")

validate(pt_fwd, cpp_fwd, "Forward : PyTorch vs C++ (raw floats)")
validate(pt_bwd, cpp_bwd, "Backward: PyTorch vs C++ (raw floats)")

print("\nDone.")