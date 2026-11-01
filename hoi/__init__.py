"""Herramientas para hipergrafos de interdependencias de alto orden (HOI)."""
from .hypergraph import select_triplets, incidence, adjacency, centralities, hub_thresholds, find_hubs, exceeds, TIE_RTOL
from .nullmodel import null_distribution
from .information import soinfo_triplets, gaussian_copula_cov
from .surrogates import phase_randomize, phase_randomize_legacy, make_surrogate
from .pipeline import compute_hoi, clean_timeseries
from .pairwise import abs_correlation, average_abs_correlation, threshold_matrices, centrality_table, pairwise_hubs, pairwise_hub_indices
from .report import hub_table, region_participation, subnetwork_share
from .group import normalize_subject, average_subjects, find_subject_files, find_significant_triplets, summarize_significance

__all__ = ["select_triplets", "incidence", "adjacency", "centralities", "hub_thresholds",
           "find_hubs", "exceeds", "TIE_RTOL", "null_distribution", "soinfo_triplets", "gaussian_copula_cov", "phase_randomize", "phase_randomize_legacy", "make_surrogate", "abs_correlation", "average_abs_correlation", "threshold_matrices", "centrality_table", "pairwise_hubs", "pairwise_hub_indices", "hub_table", "region_participation", "subnetwork_share", "normalize_subject", "find_subject_files", "average_subjects", "find_significant_triplets", "summarize_significance", "compute_hoi", "clean_timeseries"]
