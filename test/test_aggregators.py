import sys
from pathlib import Path

# allow imports when running "python scripts/..."
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.fl_aggregators import ClientUpdate, fedavg


def assert_allclose(a, b, name, atol=1e-12, rtol=1e-12):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.max(np.abs(a - b))
        raise AssertionError(f"{name} not close. max|diff|={diff}\nA={a}\nB={b}")


def test_logreg_fedavg():
    print("== test_logreg_fedavg ==")

    # two clients with different sample sizes
    u1 = ClientUpdate(
        bank="bank_a",
        n_train=100,  # weight 0.25
        params={
            "coef": np.array([[1.0, 2.0, 3.0]]),
            "intercept": np.array([1.0]),
        },
        metrics={"val_ap": 0.01, "val_n": 10},
    )
    u2 = ClientUpdate(
        bank="bank_b",
        n_train=300,  # weight 0.75
        params={
            "coef": np.array([[10.0, 20.0, 30.0]]),
            "intercept": np.array([10.0]),
        },
        metrics={"val_ap": 0.02, "val_n": 20},
    )

    out = fedavg([u1, u2])

    # expected weighted average
    w1 = 100 / (100 + 300)
    w2 = 300 / (100 + 300)

    expected_coef = w1 * u1.params["coef"] + w2 * u2.params["coef"]
    expected_intercept = w1 * u1.params["intercept"] + w2 * u2.params["intercept"]

    print("weights:", w1, w2)
    print("out coef:", out["coef"])
    print("out intercept:", out["intercept"])

    assert_allclose(out["coef"], expected_coef, "coef")
    assert_allclose(out["intercept"], expected_intercept, "intercept")

    print("[OK] logreg fedavg\n")


def test_multi_key_fedavg_like_deep():
    print("== test_multi_key_fedavg_like_deep ==")

    # simulate deep model params with two layers
    u1 = ClientUpdate(
        bank="bank_a",
        n_train=2,
        params={
            "layer1.weight": np.array([[1.0, 1.0], [1.0, 1.0]]),
            "layer2.bias": np.array([1.0, 2.0, 3.0]),
        },
        metrics={},
    )
    u2 = ClientUpdate(
        bank="bank_b",
        n_train=6,
        params={
            "layer1.weight": np.array([[9.0, 9.0], [9.0, 9.0]]),
            "layer2.bias": np.array([9.0, 8.0, 7.0]),
        },
        metrics={},
    )

    out = fedavg([u1, u2])

    w1 = 2 / (2 + 6)
    w2 = 6 / (2 + 6)

    exp_w = w1 * u1.params["layer1.weight"] + w2 * u2.params["layer1.weight"]
    exp_b = w1 * u1.params["layer2.bias"] + w2 * u2.params["layer2.bias"]

    print("weights:", w1, w2)
    print("out layer1.weight:\n", out["layer1.weight"])
    print("out layer2.bias:", out["layer2.bias"])

    assert_allclose(out["layer1.weight"], exp_w, "layer1.weight")
    assert_allclose(out["layer2.bias"], exp_b, "layer2.bias")

    print("[OK] multi-key (deep-like) fedavg\n")


def test_edge_cases():
    print("== test_edge_cases ==")

    # 1 client => output should equal that client's params
    u = ClientUpdate(
        bank="bank_a",
        n_train=5,
        params={"coef": np.array([[1.0, 2.0]]), "intercept": np.array([0.5])},
        metrics={},
    )
    out = fedavg([u])
    assert_allclose(out["coef"], u.params["coef"], "single client coef")
    assert_allclose(out["intercept"], u.params["intercept"], "single client intercept")
    print("[OK] single-client case")

    # empty updates => should raise
    try:
        fedavg([])
        raise AssertionError("Expected ValueError for empty updates, but no error raised.")
    except ValueError:
        print("[OK] empty-updates raises ValueError")

    print()


def main():
    test_logreg_fedavg()
    test_multi_key_fedavg_like_deep()
    test_edge_cases()
    print("ALL TESTS PASSED.")


if __name__ == "__main__":
    main()
