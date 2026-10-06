import os
import sys
import numpy as np
import pandas as pd

# Add src to sys.path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage2.stage_ii import compute_segmentation, plot_regimes

def test_stage2_segmentation():
    """
    Regression test to ensure Stage 2 segmentation correctly outputs
    canonical regime labels and plot_regimes runs without KeyError.
    """
    # Create mock data mimicking the output of stage_i
    t = np.linspace(0, 10, 300)
    # create a step-like delta phi that will definitely trigger all three regimes
    # stable < Q25, pre-instability between Q25 and Q75, instability > Q75
    # Since Q25 and Q75 are computed dynamically, we just use a linear ramp
    # to guarantee a nice distribution of values.
    delta_phi = np.linspace(0, 100, 300)
    E = np.ones_like(t)
    I = np.ones_like(t)
    C = np.ones_like(t)

    df = pd.DataFrame({
        "t": t,
        "DeltaPhi": delta_phi,
        "E": E,
        "I": I,
        "C": C
    })

    # 1. Run compute_segmentation
    result, q_low_val, q_high_val = compute_segmentation(df)

    # 2. Check canonical regime labels
    assigned_regimes = set(result["regime"].unique())
    canonical_regimes = {"stable", "pre-instability", "instability"}

    assert assigned_regimes.issubset(canonical_regimes), \
        f"Found invalid regimes: {assigned_regimes - canonical_regimes}"

    assert "pre-instability" in assigned_regimes, \
        "Expected 'pre-instability' to be assigned, but it wasn't."

    # 3. Check that the mathematical logic holds
    # We can inspect the smooth_DeltaPhi and regime assignment
    for _, row in result.iterrows():
        r = row["regime"]
        sdp = row["smooth_DeltaPhi"]
        if sdp < q_low_val:
            assert r == "stable"
        elif sdp > q_high_val:
            assert r == "instability"
        else:
            assert r == "pre-instability"

    # 4. Check plot_regimes runs without KeyError
    # Provide a dummy path
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy_stage2_plot.png")
    try:
        plot_regimes(result, q_low_val, q_high_val, path=dummy_path)
    except Exception as e:
        assert False, f"plot_regimes raised an exception: {e}"

    # cleanup dummy file
    if os.path.exists(dummy_path):
        os.remove(dummy_path)

if __name__ == "__main__":
    test_stage2_segmentation()
    print("All tests passed.")