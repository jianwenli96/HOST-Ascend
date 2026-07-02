
import torch

def test_cdist_grad():
    print("Testing cdist gradient at 0 distance...")
    v1 = torch.randn(1, 4, 16, requires_grad=True, dtype=torch.float32)
    v2 = v1.clone() # Identical
    
    dist = torch.cdist(v1, v2, p=2)
    loss = dist.sum()
    loss.backward()
    
    print(f"v1.grad contains NaN: {torch.isnan(v1.grad).any()}")
    print(v1.grad)

def test_sq_dist_grad():
    print("\nTesting expanded squared dist gradient at 0 distance...")
    v1 = torch.randn(1, 4, 16, requires_grad=True, dtype=torch.float32)
    v2 = v1.clone()
    
    x2 = (v1**2).sum(-1, keepdim=True)
    y2 = (v2**2).sum(-1, keepdim=True).transpose(1, 2)
    xy = torch.bmm(v1, v2.transpose(1,2))
    dist_sq = x2 + y2 - 2*xy
    
    loss = dist_sq.sum()
    loss.backward()
    
    print(f"v1.grad contains NaN: {torch.isnan(v1.grad).any()}")
    # Gradient of |x-x|^2 is 0.

if __name__ == "__main__":
    test_cdist_grad()
    test_sq_dist_grad()
