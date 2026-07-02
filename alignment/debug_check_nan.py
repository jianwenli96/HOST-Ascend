
import torch
from utils import check_nan

def test_check_nan():
    print("Testing check_nan with -Inf...")
    t = torch.tensor([-float('inf')])
    try:
        check_nan(t, "test_tensor", "main")
        print("check_nan PASSED for -Inf (Unexpected if isinf check exists)")
    except ValueError as e:
        print(f"check_nan CAUGHT -Inf: {e}")

    print("\nTesting check_nan with NaN...")
    t = torch.tensor([float('nan')])
    try:
        check_nan(t, "test_tensor", "main")
        print("check_nan PASSED for NaN")
    except ValueError as e:
        print(f"check_nan CAUGHT NaN: {e}")

if __name__ == "__main__":
    # Mock CONFIG for utils
    import config
    config.CONFIG.DEBUG = True
    test_check_nan()
