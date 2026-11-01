"""Comprueba que el paquete `hoi` reproduce el código original.

Variables de entorno (los tests que las necesitan se omiten si faltan):
  HOI_DATA_DIR   carpeta con average_triplets.csv y aggregated_centralities1000copies0005fraction.csv
  HOI_LEGACY_DIR carpeta con HOI_connectivity.py e infotopo_server.py
Ejecutar:  python -m pytest tests   (o)   python tests/test_equivalence.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import hoi  # noqa: E402
import legacy_hypergraph as legacy  # noqa: E402
import legacy_block3 as leg3  # noqa: E402
import legacy_block5 as leg5  # noqa: E402

EXAMPLE = os.environ.get("HOI_EXAMPLE_DIR")   # carpeta con AAL_timeseries_example.txt y example_triplets.csv (tarda ~3 min)
DATA = os.environ.get("HOI_DATA_DIR")
LEGACY = os.environ.get("HOI_LEGACY_DIR")


def _need(path_var, what):
    if not path_var:
        raise unittest.SkipTest(f"define {what}")


class TestHypergraph(unittest.TestCase):
    def test_adjacency_and_centralities_match_legacy(self):
        _need(DATA, "HOI_DATA_DIR")
        df = pd.read_csv(os.path.join(DATA, "average_triplets.csv"), index_col=0)
        for mode in ("redundancy", "synergy"):
            top_old, H_old = legacy.create_hypergraph(df, mode=mode)
            cent_old = legacy.HO_cent_df(H_old, top_old)
            top = hoi.select_triplets(df, mode=mode)
            A = hoi.adjacency(top["nplets"])
            self.assertTrue([tuple(t) for t in top_old["nplets"]] == list(top["nplets"]))
            self.assertTrue((A == H_old).all(), mode)
            cent = hoi.centralities(A, top["nplets"])
            for c in ("EC", "BC", "DC", "EC_z", "BC_z", "DC_z"):
                np.testing.assert_allclose(cent[c], cent_old[c], atol=1e-9, err_msg=f"{mode} {c}")

    def test_hub_counts_regression(self):
        _need(DATA, "HOI_DATA_DIR")
        df = pd.read_csv(os.path.join(DATA, "average_triplets.csv"), index_col=0)
        null = pd.read_csv(os.path.join(DATA, "aggregated_centralities1000copies0005fraction.csv"))
        thr = hoi.hub_thresholds(null, p=0.05)
        expected = {("Mut Info_normalized", "redundancy"): 140, ("Mut Info_normalized", "synergy"): 219,
                    ("Total Corr_zscore", "redundancy"): 80, ("Total Corr_zscore", "synergy"): 150}
        for (col, mode), n_hubs in expected.items():
            top = hoi.select_triplets(df, column=col, mode=mode)
            cent = hoi.centralities(hoi.adjacency(top["nplets"]), top["nplets"])
            self.assertEqual(len(hoi.find_hubs(cent, thr, "EC", use_z=True)), n_hubs, (col, mode))


class TestInformation(unittest.TestCase):
    def test_soinfo_matches_legacy(self):
        _need(LEGACY, "HOI_LEGACY_DIR")
        sys.path.insert(0, LEGACY)
        import HOI_connectivity as old
        rng = np.random.default_rng(0)
        N, T = 12, 400
        mix = rng.standard_normal((N, N))
        data = mix @ rng.standard_normal((N, T))            # (N, T), regiones x tiempo
        _, _, oinfo_old, sinfo_old, nplets_old = old.high_order(data, 3)
        cov = hoi.gaussian_copula_cov((data - data.mean(1, keepdims=True)).T)
        nplets, oinfo, sinfo = hoi.soinfo_triplets(cov, T, n=3)
        self.assertTrue((nplets == nplets_old).all())
        np.testing.assert_allclose(oinfo, oinfo_old, atol=1e-9)
        np.testing.assert_allclose(sinfo, sinfo_old, atol=1e-9)


class TestBlock1(unittest.TestCase):
    def test_compute_hoi_matches_legacy_run_all_HOI(self):
        _need(LEGACY, "HOI_LEGACY_DIR")
        sys.path.insert(0, LEGACY)
        import HOI_connectivity as old
        import infotopo_server
        N, T = 10, 300
        rng = np.random.default_rng(5)
        ts = pd.DataFrame(rng.standard_normal((T, N)) @ rng.standard_normal((N, N)))
        ts.loc[T] = np.nan                                   # fila final con NaN, como en los archivos de HCP
        old.atlas_size = N                                   # el original tiene 92 fijo a nivel de módulo
        old.information_topo = infotopo_server.infotopo(dimension_max=3, dim_to_rank=2, number_of_max_val=5,
                                                        dimension_tot=N, nb_of_values=20, forward_computation_mode=True)
        ref = old.run_all_HOI(ts.iloc[:-1, :])
        new = hoi.compute_hoi(ts)
        self.assertEqual(list(ref.columns), list(new.columns))
        self.assertEqual(len(ref), len(new))
        self.assertTrue([list(x) for x in ref["nplets"]] == [list(x) for x in new["nplets"]])
        for c in ("Oinfo", "Sinfo", "Joint Ent", "Mut Info", "Total Corr"):
            np.testing.assert_allclose(new[c].astype(float), ref[c].astype(float), atol=1e-9, err_msg=c)


class TestReorganizedModule(unittest.TestCase):
    def _load_new(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("new_hc", os.path.join(HERE, "..", "HOI_connectivity.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_high_order_red_syn_match_legacy(self):
        _need(LEGACY, "HOI_LEGACY_DIR")
        sys.path.insert(0, LEGACY)
        import HOI_connectivity as old
        new = self._load_new()
        data = np.random.default_rng(2).standard_normal((9, 9)) @ np.random.default_rng(3).standard_normal((9, 250))
        Ro, So, Oo, Sio, No = old.high_order(data, 3)
        Rn, Sn, On, Sin, Nn = new.high_order(data, 3)
        self.assertTrue((No == Nn).all())
        for a, b in ((Ro, Rn), (So, Sn), (Oo, On), (Sio, Sin)):
            np.testing.assert_allclose(np.array(b, float), np.array(a, float), atol=1e-9, equal_nan=True)

    def test_real_example_matches_saved_output(self):
        """Serie real de HCP del repo. Entropías: idénticas (~1e-16). O/S-info: el CSV guardado difiere ~1e-5 del
        código original ejecutado hoy (otra versión de librerías); el nuevo coincide con el original hasta ~1e-15."""
        _need(EXAMPLE, "HOI_EXAMPLE_DIR")
        import ast
        ts = pd.read_csv(os.path.join(EXAMPLE, "AAL_timeseries_example.txt"), sep="\t", header=None)
        ref = pd.read_csv(os.path.join(EXAMPLE, "example_triplets.csv"), index_col=0)
        new = hoi.compute_hoi(ts)
        self.assertEqual(list(new.columns), list(ref.columns))
        self.assertTrue(all(list(a) == ast.literal_eval(b) for a, b in zip(new["nplets"], ref["nplets"])))
        for c in ("Joint Ent", "Mut Info", "Total Corr"):
            np.testing.assert_allclose(new[c].astype(float), ref[c].astype(float), atol=1e-12, err_msg=c)
        for c in ("Oinfo", "Sinfo"):
            np.testing.assert_allclose(new[c].astype(float), ref[c].astype(float), atol=1e-4, err_msg=c)


class TestPlotsAndTies(unittest.TestCase):
    def test_figures_build(self):
        import matplotlib
        matplotlib.use("Agg")
        from hoi import atlas as A, plots
        atlas = A.load_atlas()
        self.assertEqual(len(atlas["names"]), 92)
        rng = np.random.default_rng(0)
        tri = np.array([rng.choice(92, 3, replace=False) for _ in range(60)])
        hub = rng.random(60) < .3
        for fig in (plots.plot_hub_projections(atlas, tri, hub, A.load_brain()), plots.plot_pair_matrix(atlas, tri[hub]),
                    plots.plot_null_vs_real(rng.normal(size=500), rng.normal(1, 1, 60), 1.5),
                    plots.plot_pairwise_vs_highorder(atlas, [26, 60, 21], tri[hub])):
            self.assertTrue(len(fig.axes) >= 1)

    def test_ties_with_threshold_do_not_count(self):
        thr = 3 / 626
        vals = np.array([3 * (1 / 626), thr, 4 / 626, 6.5e-15 * 1.0000001])
        self.assertEqual(hoi.exceeds(vals, thr).tolist(), [False, False, True, False])      # el último < umbral
        self.assertTrue(hoi.exceeds(np.array([1.8e-13]), 6.5e-15)[0])                       # umbral ~0: no se pierde

class TestBlock3(unittest.TestCase):
    @staticmethod
    def _subjects(tmp, n=4, M=3000, seed=0, prefix="s"):
        """CSV por sujeto con el mismo formato que las salidas del bloque 1 (índice + 6 columnas, nplets como texto)."""
        rng = np.random.default_rng(seed)
        paths = []
        for k in range(n):
            df = pd.DataFrame({c: rng.standard_normal(M) * (k + 1) + 3 * k for c in ["Oinfo", "Sinfo", "Joint Ent", "Mut Info", "Total Corr"]})
            df["nplets"] = [str([i, i + 1, i + 2]) for i in range(M)]
            p = os.path.join(tmp, f"{prefix}{k}.csv"); df.to_csv(p); paths.append(p)
        return paths

    def test_average_matches_legacy(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._subjects(tmp)
            old = leg3.legacy_average([pd.read_csv(p, index_col=0) for p in paths])
            new = hoi.average_subjects(paths)
        self.assertEqual(list(old.columns), list(new.columns))
        self.assertEqual(old["nplets"].tolist(), new["nplets"].tolist())
        np.testing.assert_allclose(new.drop(columns="nplets").to_numpy(), old.drop(columns="nplets").to_numpy(), atol=1e-12)

    def test_average_detects_wrong_order_and_lfs_pointer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._subjects(tmp, n=2)
            bad = pd.read_csv(paths[1], index_col=0).iloc[::-1].reset_index(drop=True)
            bad.to_csv(paths[1])
            with self.assertRaisesRegex(ValueError, "mismo orden"):
                hoi.average_subjects(paths)
            ptr = os.path.join(tmp, "ptr.csv")
            Path_ = open(ptr, "w"); Path_.write("version https://git-lfs.github.com/spec/v1\noid sha256:6088f677\nsize 14760171\n"); Path_.close()
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                hoi.average_subjects([ptr])

    def test_missing_folder_error_says_where_it_looked(self):
        with self.assertRaises(FileNotFoundError) as cm:
            hoi.find_subject_files("../Data/Real/Results_RL/*.csv")
        msg = str(cm.exception)
        self.assertIn("Se ha buscado en", msg)
        self.assertIn("averages", msg)

    def test_significant_matches_legacy_on_real_averages(self):
        _need(DATA, "HOI_DATA_DIR")
        a = pd.read_csv(os.path.join(DATA, "average_triplets.csv"), index_col=0)
        b = pd.read_csv(os.path.join(DATA, "average_triplets_1.csv"), index_col=0)   # solo como segundo conjunto de datos reales
        for col in ("Mut Info", "Oinfo", "Total Corr"):
            for metric in ("zscore", "normalized"):
                ref = leg3.legacy_find_significant_triplets(a, b, col, metric)
                new = hoi.find_significant_triplets(a, b, col, metric)
                pd.testing.assert_frame_equal(new, ref)


class TestBlock3Real(unittest.TestCase):
    def test_reproduces_counts_saved_in_the_notebook(self):
        """average_triplets_1.csv (carpeta CodeBlock3) + average_triplets_random.csv: el notebook guarda 68458 = 28685 + 39773."""
        _need(DATA, "HOI_DATA_DIR")
        real = pd.read_csv(os.path.join(DATA, "average_triplets_1.csv"), index_col=0)
        rnd = pd.read_csv(os.path.join(DATA, "average_triplets_random.csv"), index_col=0)
        sig = hoi.find_significant_triplets(real, rnd, "Mut Info", metric="zscore")
        self.assertEqual((len(sig), int((sig["Category"] == "Redundant").sum()), int((sig["Category"] == "Synergistic").sum())),
                         (68458, 28685, 39773))


class TestBlock4Real(unittest.TestCase):
    """Contra las cifras que el notebook ORIGINAL del bloque 4 tiene guardadas en sus salidas."""
    REF_THR = {"EC": 6.533731412543565e-15, "EC_z": -0.0936633620275967, "BC": 0.0003833865814696,
               "BC_z": 1.4067675976534844, "DC": 0.0047923322683706, "DC_z": 1.6603495340209775}
    REF_RED = {"EC": [8.931766e-03, 2.484731e-03, 8.062214e-03, 7.991820e-03, 4.573218e-03], "DC": [0.092652, 0.067093, 0.086262, 0.076677, 0.071885],
               "EC_z": [-0.163925, -0.338111, -0.187419, -0.189321, -0.281684], "DC_z": [3.759480, 2.033704, 3.328036, 2.680870, 2.357287]}
    REF_TC = {"EC": [1.140368e-01, 1.129427e-01, 1.122630e-01, 4.931379e-03, 1.121284e-01], "EC_z": [2.676012, 2.646423, 2.628040, -0.274790, 2.624400]}

    def test_thresholds_and_centralities_match_notebook_outputs(self):
        _need(DATA, "HOI_DATA_DIR")
        null = pd.read_csv(os.path.join(DATA, "aggregated_centralities1000copies0005fraction.csv"))
        thr = hoi.hub_thresholds(null, 0.05)
        for k, v in self.REF_THR.items():
            self.assertAlmostEqual(thr[k] / v, 1.0, places=8, msg=k)
        df = pd.read_csv(os.path.join(DATA, "average_triplets.csv"), index_col=0)     # el que usa el bloque 4 (no average_triplets_1.csv)
        for col, ref in (("Mut Info_normalized", self.REF_RED), ("Total Corr_zscore", self.REF_TC)):
            top = hoi.select_triplets(df, col, "redundancy")
            cent = hoi.centralities(hoi.adjacency(top["nplets"]), top["nplets"])
            for k, v in ref.items():
                np.testing.assert_allclose(cent[k].iloc[:5].to_numpy(), v, rtol=2e-4, atol=1e-7, err_msg=f"{col} {k}")
        top = hoi.select_triplets(df, "Mut Info_normalized", "redundancy")
        cent = hoi.centralities(hoi.adjacency(top["nplets"]), top["nplets"])
        self.assertEqual(int(hoi.exceeds(cent["EC"], thr["EC"]).sum()), 627)           # el notebook imprime 627


class TestPaperV1(unittest.TestCase):
    """Reproduce lo que dice el manuscrito v1 (bioRxiv 2023.02.10.528083v1): 1000 tripletes; 42 hubs de II (redundancia) que comparten
    la precentral y la postcentral izquierdas; hubs de TC en ambos cuneus; hubs sinérgicos centrados en el giro angular izquierdo."""
    def _hubs(self, col, mode, thr):
        from hoi import atlas as A
        names = A.load_atlas()["names"]
        df = pd.read_csv(os.path.join(DATA, "average_triplets.csv"), index_col=0)     # el conjunto del artículo (con _1 salen 41)
        top = hoi.select_triplets(df, col, mode, n_triplets=1000)
        cent = hoi.centralities(hoi.adjacency(top["nplets"]), top["nplets"])
        tri = np.array(top["nplets"].tolist())[(cent["EC"] > thr).to_numpy()] - 1
        return tri, {n: names.index(n) for n in ("Precentral_L", "Postcentral_L", "Cuneus_L", "Cuneus_R", "Angular_L")}

    def test_n_triplets_overrides_fraction(self):
        _need(DATA, "HOI_DATA_DIR")
        df = pd.read_csv(os.path.join(DATA, "average_triplets.csv"), index_col=0)
        self.assertEqual(len(hoi.select_triplets(df, n_triplets=1000, fraction=0.9)), 1000)

    def test_interaction_information_42_hubs_in_pre_and_postcentral_left(self):
        _need(DATA, "HOI_DATA_DIR")
        tri, ix = self._hubs("Mut Info_normalized", "redundancy", 0.12)
        self.assertEqual(len(tri), 42)
        self.assertTrue(all(ix["Precentral_L"] in t and ix["Postcentral_L"] in t for t in tri))

    def test_total_correlation_hubs_share_both_cuneus(self):
        _need(DATA, "HOI_DATA_DIR")
        tri, ix = self._hubs("Total Corr_zscore", "redundancy", 0.05)
        self.assertGreater(len(tri), 0)
        self.assertTrue(all(ix["Cuneus_L"] in t and ix["Cuneus_R"] in t for t in tri))

    def test_synergy_hubs_centered_on_left_angular(self):
        _need(DATA, "HOI_DATA_DIR")
        tri, ix = self._hubs("Mut Info_normalized", "synergy", 0.05)
        self.assertGreater(len(tri), 0)
        self.assertTrue(all(ix["Angular_L"] in t for t in tri))

    def test_null_distribution_with_fixed_size(self):
        _need(DATA, "HOI_DATA_DIR")
        df = pd.read_csv(os.path.join(DATA, "average_triplets.csv"), index_col=0)
        self.assertEqual(len(hoi.null_distribution(df, n_copies=2, workers=2, n_triplets=1000)), 2000)


class TestBlock5(unittest.TestCase):
    @staticmethod
    def _subjects(seed=0, n=3, T=300, N=30, structured=True):
        """Series con estructura de bloques (3 redes de distinta fuerza) para que haya un vector propio dominante claro.
        structured=False: mezcla aleatoria sin estructura (la iteración de potencias del original puede no converger)."""
        rng = np.random.default_rng(seed)
        mix = rng.standard_normal((N, N))
        out = []
        for _ in range(n):
            if structured:
                F = rng.standard_normal((T, 3))
                block = np.repeat([0, 1, 2], N // 3)
                load = np.repeat([1.6, 1.0, 0.6], N // 3) + 0.15 * rng.random(N)
                X = F[:, block] * load + rng.standard_normal((T, N))
            else:
                X = rng.standard_normal((T, N)) @ mix
            df = pd.DataFrame(X)
            df.loc[T] = np.nan                                  # fila final de NaN, como los archivos de HCP
            out.append(df)
        return out

    def _compare(self, subjects):
        old = leg5.legacy_average(subjects)
        new = np.mean([hoi.abs_correlation(s) for s in subjects], axis=0)
        np.testing.assert_allclose(new, old.values, atol=1e-12)
        crit, b, w = leg5.legacy_matrices(old)
        m = hoi.threshold_matrices(new)
        self.assertAlmostEqual(m["critical"], crit, places=12)
        self.assertTrue((m["binarized"] == b.values).all())
        np.testing.assert_allclose(m["weighted"], w.values, atol=1e-12)
        lh, t = leg5.legacy_hubs(b, w), hoi.centrality_table(m["binarized"], m["weighted"])
        for k in ("EC", "DC", "BC", "EC_w", "BC_w"):
            mine = t[t[k + "_hub"]].sort_values(k + "_z", ascending=False).index.tolist()
            self.assertEqual(lh[k]["Node"].tolist(), mine, k)

    def test_matches_legacy_on_synthetic_subjects(self):
        self._compare(self._subjects())

    def test_matches_legacy_on_real_series_segments(self):
        _need(EXAMPLE, "HOI_EXAMPLE_DIR")
        ts = pd.read_csv(os.path.join(EXAMPLE, "AAL_timeseries_example.txt"), sep="\t", header=None)
        seg = [ts.iloc[i * 398:(i + 1) * 398].reset_index(drop=True) for i in range(3)]
        for s_ in seg:
            s_.loc[len(s_)] = np.nan
        self._compare(seg)

    def test_ec_works_where_power_iteration_fails(self):
        """Sin estructura dominante, nx.eigenvector_centrality (100 iteraciones) puede lanzar PowerIterationFailedConvergence; el nuevo no."""
        import networkx as nx
        avg = np.mean([hoi.abs_correlation(s) for s in self._subjects(0, structured=False)], axis=0)
        A = hoi.threshold_matrices(avg)["binarized"]
        ec = hoi.centrality_table(A)["EC"].to_numpy()
        self.assertTrue(np.isfinite(ec).all())
        lam = np.linalg.eigvalsh(A.astype(float))[-1]
        np.testing.assert_allclose(A @ ec, lam * ec, atol=1e-9)                       # es realmente un vector propio dominante
        try:
            ref = np.array(list(nx.eigenvector_centrality(nx.from_numpy_array(A.astype(float))).values()))
            np.testing.assert_allclose(ec, ref, atol=1e-5)                              # si el original converge, coincide
        except nx.PowerIterationFailedConvergence:
            pass                                                                        # el original falla; el nuevo sigue funcionando

    def test_self_loops_alone_do_not_change_centralities(self):
        avg = np.mean([hoi.abs_correlation(s) for s in self._subjects(1)], axis=0)
        A0 = hoi.threshold_matrices(avg, include_diagonal=False)["binarized"]
        A1 = A0.copy()
        np.fill_diagonal(A1, 1)
        t0, t1 = hoi.centrality_table(A0), hoi.centrality_table(A1)
        for c in ("EC", "DC_z", "BC"):
            np.testing.assert_allclose(t1[c], t0[c], atol=1e-9)

    def test_including_diagonal_raises_threshold(self):
        avg = np.mean([hoi.abs_correlation(s) for s in self._subjects(2)], axis=0)
        self.assertGreater(hoi.threshold_matrices(avg, include_diagonal=True)["critical"],
                           hoi.threshold_matrices(avg, include_diagonal=False)["critical"])

    def test_atlas_names_are_not_truncated(self):
        from hoi import atlas as A
        names = A.load_atlas()["names"]
        self.assertIn("Frontal_Inf_Tri_L", names)
        self.assertIn("Precuneus_L", names)


class TestReport(unittest.TestCase):
    def test_hub_table_and_participation(self):
        from hoi import atlas as A
        atlas = A.load_atlas()
        tri = np.array([[0, 1, 2], [0, 1, 3], [10, 11, 12]])
        cent = pd.DataFrame({"EC": [.3, .2, .1], "BC": 0, "DC": 0, "EC_z": [3., 2., 1.], "BC_z": 0, "DC_z": 0})
        t = hoi.hub_table(atlas, cent, [True, True, False], measure="EC_z", triplets=tri)
        self.assertEqual(len(t), 2)
        self.assertEqual(t.loc[0, "region_1"], atlas["names"][0])
        p = hoi.region_participation(atlas, tri[:2])
        self.assertEqual(int(p.set_index("region").loc[atlas["names"][0], "hubs"]), 2)
        self.assertAlmostEqual(float(hoi.subnetwork_share(atlas, tri[:2]).sum()), 100, delta=0.2)


class TestSurrogates(unittest.TestCase):
    @staticmethod
    def _series(T=1200, N=6, seed=1):
        """Series con autocorrelación y correlación entre regiones MODERADA (~0,3-0,5), como las reales."""
        rng = np.random.default_rng(seed)
        common = np.convolve(rng.standard_normal(T + 9), np.ones(10) / 10, "valid")
        own = np.stack([np.convolve(rng.standard_normal(T + 9), np.ones(10) / 10, "valid") for _ in range(N)], 1)
        return 1000 + 50 * (0.6 * common[:, None] + own)

    def test_spectrum_mean_and_autocorrelation_preserved(self):
        x = self._series()
        for indep in (True, False):
            y = hoi.phase_randomize(x, rng=2, independent=indep)
            np.testing.assert_allclose(np.abs(np.fft.rfft(y, axis=0)), np.abs(np.fft.rfft(x, axis=0)), rtol=1e-9)
            np.testing.assert_allclose(y.mean(0), x.mean(0), rtol=1e-12)
            self.assertFalse(np.allclose(x, y))

    def test_shared_preserves_correlation_matrix_exactly(self):
        x = self._series()
        y = hoi.phase_randomize(x, rng=3, independent=False)
        self.assertGreater(np.abs(np.corrcoef(x.T)[np.triu_indices(6, 1)]).mean(), 0.15)   # correlación no trivial
        np.testing.assert_allclose(np.corrcoef(y.T), np.corrcoef(x.T), atol=1e-9)

    def test_independent_destroys_correlation_between_regions(self):
        x = self._series()
        y = hoi.phase_randomize(x, rng=3, independent=True)
        r = lambda m: np.abs(np.corrcoef(m.T)[np.triu_indices(6, 1)]).mean()
        self.assertLess(r(y), 0.5 * r(x))

    def test_make_surrogate_keeps_shape_columns_and_drops_nan_rows(self):
        df = pd.DataFrame(self._series(), columns=list("abcdef"))
        df.loc[len(df)] = np.nan
        out = hoi.make_surrogate(df, "independent", seed=0)
        self.assertEqual(out.shape, (len(df) - 1, 6))
        self.assertEqual(list(out.columns), list("abcdef"))
        np.testing.assert_array_equal(out.to_numpy(), hoi.make_surrogate(df, "independent", seed=0).to_numpy())   # reproducible

    def test_legacy_reproduces_repo_example_statistics(self):
        _need(EXAMPLE, "HOI_EXAMPLE_DIR")
        df = pd.read_csv(os.path.join(EXAMPLE, "example.txt"), sep="\t", header=None).dropna()
        pr = pd.read_csv(os.path.join(EXAMPLE, "example_pr.csv"), index_col=0).to_numpy(float)
        leg = hoi.make_surrogate(df, "legacy", seed=0).to_numpy(float)
        ac = lambda m: np.mean([np.corrcoef(m[:-1, j], m[1:, j])[0, 1] for j in range(m.shape[1])])
        rr = lambda m: np.abs(np.corrcoef(m.T)[np.triu_indices(m.shape[1], 1)]).mean()
        self.assertAlmostEqual(ac(leg), ac(pr), delta=0.05)
        self.assertAlmostEqual(rr(leg), rr(pr), delta=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
