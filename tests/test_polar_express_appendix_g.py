import math

from polar.polynomial.express import additive_appendix_g_composition, paper_polar_express_coeff


def test_appendix_g_additive_reproduces_paper_coeffs() -> None:
    steps = additive_appendix_g_composition(1.0e-3, 8)

    for idx, step in enumerate(steps):
        ref = paper_polar_express_coeff(idx)
        assert math.isclose(step.a, ref.a, rel_tol=0.0, abs_tol=5e-9)
        assert math.isclose(step.b, ref.b, rel_tol=0.0, abs_tol=5e-9)
        assert math.isclose(step.c, ref.c, rel_tol=0.0, abs_tol=5e-9)


def test_appendix_g_additive_improves_predicted_condition_number() -> None:
    steps = additive_appendix_g_composition(1.0e-1, 3)

    assert steps[0].pred_kappa_after < 10.0
    assert steps[1].pred_kappa_after <= steps[0].pred_kappa_after
    assert steps[2].pred_kappa_after <= steps[1].pred_kappa_after
