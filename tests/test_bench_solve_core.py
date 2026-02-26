from benchmarks.solve.bench_solve_core import matrix_solve_methods


def test_matrix_solve_methods_p1_excludes_chebyshev_and_keeps_torch_baselines():
    methods = matrix_solve_methods(1)
    assert "Chebyshev-Apply" not in methods
    assert "Torch-Solve" in methods
    assert "Torch-Cholesky-Solve" in methods
    assert "Torch-Cholesky-Solve-ReuseFactor" in methods


def test_matrix_solve_methods_p2_includes_chebyshev():
    methods = matrix_solve_methods(2)
    assert "Chebyshev-Apply" in methods


def test_matrix_solve_methods_p4_includes_chebyshev():
    methods = matrix_solve_methods(4)
    assert "Chebyshev-Apply" in methods
