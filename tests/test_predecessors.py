"""The predecessor sets: structure, counts, seam, cross-time blocks."""
import numpy as np
import pytest

from flowsmoother.joint import build_filter_graph, build_joint_graph, check_graph
from flowsmoother.rows import assemble


def interior(band, r):
    return np.where((band.cell_i >= r) & (band.cell_i < band.nr - r) &
                    (band.cell_j >= r) & (band.cell_j < band.nc - r))[0]


@pytest.mark.parametrize("r", [1, 2, 3])
def test_filter_graph_invariants_and_counts(band, r):
    g = build_filter_graph(band, r)
    assert check_graph(g, band)
    K = (2 * r + 1) ** 2
    half = (K - 1) // 2
    inner = interior(band, r)
    assert np.all(g.groups["f_u"].plen[inner] == half)             # half square of u
    assert np.all(g.groups["f_v"].plen[inner] == K + half)         # all u, half v
    assert np.all(g.groups["f_h"].plen[inner] == 2 * K + half)     # all u, v, half h


@pytest.mark.parametrize("r", [1, 2])
def test_every_neighbour_pair_is_linked_exactly_once(band, r):
    """Same-field links = unordered pairs at Chebyshev distance <= r, once each."""
    g = build_filter_graph(band, r)
    m = band.m
    grp = g.groups["f_u"]
    links = {(int(t), int(p)) for t, row, n in zip(grp.targets, grp.pred, grp.plen)
             for p in row[:n]}
    expect = set()
    for c in range(m):
        d = band.chebyshev(np.full(m, c), np.arange(m))
        for c2 in np.where((d <= r) & (np.arange(m) < c))[0]:
            expect.add((c, int(c2)))
    assert links == expect


def test_seam_is_handled_by_index_order(band):
    """Across the periodic seam the higher index holds the lower as predecessor."""
    g = build_filter_graph(band, 1)
    i = 5
    first, last = i * band.nc, i * band.nc + band.nc - 1          # (i, 0) and (i, nc-1)
    assert first in g.predecessors(last)
    assert last not in g.predecessors(first)
    assert (i - 1) * band.nc + band.nc - 1 in g.predecessors(first)   # wrapped NW neighbour


def test_cross_field_links_are_full_squares(band):
    g = build_filter_graph(band, 2)
    c = int(interior(band, 2)[7])
    sq = band.square(2)[c]
    ph = g.predecessors(2 * band.m + c)
    assert set(sq) <= set(ph)                        # u of the whole square, own cell too
    assert set(band.m + sq) <= set(ph)               # v of the whole square
    assert c in ph and band.m + c in ph              # u and v at the cell itself


@pytest.mark.parametrize("r", [1, 2])
def test_joint_graph_blocks(band, r):
    m, n = band.m, band.dim
    cross = band.square(r)
    g = build_joint_graph(band, r, cross_h=cross)
    assert check_graph(g, band)
    f = build_filter_graph(band, r)
    for fname in "uvh":
        a, b, flt = g.groups["a_" + fname], g.groups["b_" + fname], f.groups["f_" + fname]
        # x_a rows are exactly the filter rows; nothing from x_b
        assert np.array_equal(a.plen, flt.plen)
        assert np.all(a.pred[a.pred < g.dim] < n)
        # x_b rows: the same same-time set shifted by 3m, plus three fields of the square at t-lag
        for k in (0, m // 2 + 3, m - 1):
            p = b.pred[k, :b.plen[k]]
            same, cr = p[p >= n] - n, p[p < n]
            assert np.array_equal(same, flt.pred[k, :flt.plen[k]])
            cells = cross[k][cross[k] >= 0]
            assert np.array_equal(cr, np.sort(np.concatenate([q * m + cells for q in range(3)])))
            assert {k, m + k, 2 * m + k} <= set(cr)          # own cell, three fields


def test_h_rows_take_the_given_cross_cells_and_uv_rows_keep_the_square(band):
    m, n = band.m, band.dim
    rng = np.random.default_rng(0)
    cells = np.full((m, 6), -1)
    for c in range(m):
        k = rng.integers(1, 7)
        cells[c, :k] = np.sort(rng.choice(m, size=k, replace=False))
    g = build_joint_graph(band, 2, cross_h=cells)
    assert check_graph(g, band)
    sq = band.square(2)
    for c in (0, 17, m - 1):
        ph = g.predecessors(n + 2 * m + c)
        cr = ph[ph < n]
        want = cells[c][cells[c] >= 0]
        assert np.array_equal(cr, np.sort(np.concatenate([q * m + want for q in range(3)])))
        pu = g.predecessors(n + c)
        cu = pu[pu < n]
        ws = sq[c][sq[c] >= 0]
        assert np.array_equal(cu, np.sort(np.concatenate([q * m + ws for q in range(3)])))


def test_no_cross_links_when_asked(band):
    none = np.full((band.m, 1), -1)
    g = build_joint_graph(band, 1, cross_h=None, cross_uv=none)
    assert check_graph(g, band)
    for name in ("b_u", "b_v", "b_h"):
        grp = g.groups[name]
        assert np.all(grp.pred[grp.pred < g.dim] >= band.dim)


def test_pattern_of_L_is_the_graph(band):
    g = build_joint_graph(band, 1, cross_h=band.square(2))
    parts = [(q.targets, q.pred, np.where(q.pred < g.dim, 0.5, 0.0), np.ones(q.targets.size))
             for q in g.groups.values()]
    fac = assemble(g.dim, parts)
    L = fac.L.tocsr()
    assert np.allclose(L.diagonal(), 1.0)
    assert (L - L.multiply(0) != 0).nnz == L.nnz
    for t in (5, band.dim + 40, g.dim - 1):
        cols = np.sort(L[t].indices)
        assert np.array_equal(cols[cols != t], g.predecessors(t))
    assert abs(fac.mean_predecessors() - g.mean_predecessors()) < 1e-12
    assert (L.tocoo().row >= L.tocoo().col).all()               # lower triangular


def test_matched_patch_has_the_requested_size(band):
    rng = np.random.default_rng(1)
    counts = rng.integers(1, 60, size=band.m)
    patch = band.patch(counts)
    got = (patch >= 0).sum(axis=1)
    assert np.array_equal(got, counts)
    for c in (0, 50, band.m - 1):
        cells = patch[c][patch[c] >= 0]
        assert c in cells and len(set(cells)) == len(cells)
        # isotropic: no cell outside the patch is strictly nearer than one inside
        di = band.cell_i - band.cell_i[c]
        dj = (band.cell_j - band.cell_j[c] + band.nc // 2) % band.nc - band.nc // 2
        d2 = di ** 2 + dj ** 2
        outside = np.setdiff1d(np.arange(band.m), cells)
        if outside.size:
            assert d2[cells].max() <= d2[outside].min()
