import yaml
from pathlib import Path

import pytest

from run_experiment import generate_experiment_configs, load_config, validate_config


CONFIG_PATH = Path('config/uci_config.yaml')


@pytest.fixture
def base_config():
    return load_config(str(CONFIG_PATH))


def test_config_has_required_sections(base_config):
    assert validate_config(base_config) is True
    assert 'data' in base_config
    assert 'model' in base_config
    assert 'paths' in base_config


def test_segment_and_increment_dims_match(base_config):
    assert len(base_config['data']['segment_dims']) == len(base_config['data']['increment_dims'])


def test_full_mode_generates_expected_experiment_count(base_config):
    config = base_config.copy()
    config['test_mode'] = False
    configs = generate_experiment_configs(config)

    n_channels = len(config['data']['channel_configs'])
    n_windows = len(config['data']['segment_dims'])
    n_models = len(config['model']['types'])

    assert len(configs) == n_channels * n_windows * n_models


def test_test_mode_generates_single_experiment(base_config):
    config = base_config.copy()
    config['test_mode'] = True
    configs = generate_experiment_configs(config)
    assert len(configs) == 1
    assert configs[0][0].startswith('TEST_')


def test_time_domain_feature_count():
    from emg_classifier.features.uci_extractor import UCIFeatureExtractor

    config = yaml.safe_load(CONFIG_PATH.read_text())
    config['segment_dim'] = 256
    config['increment_dim'] = 64
    config['feature_dim'] = 104
    config['channel_no'] = 8
    config['channel_subset'] = list(range(1, 9))

    extractor = UCIFeatureExtractor(config)
    import numpy as np

    sample = np.random.randn(256)
    time_features = extractor._extract_time_domain_features(sample)
    ar_features = extractor._extract_ar_coefficients(sample)

    assert len(time_features) == 7
    assert len(ar_features) == 6
    assert len(time_features) + len(ar_features) == 13


def test_windows_do_not_cross_session_boundaries():
    """Regression test: sliding windows must never blend two different recording
    sessions/files. Feature extraction windows within each session_id independently,
    so the number of windows produced must equal the sum of per-session window
    counts, not the window count you'd get from treating the concatenated data as
    one contiguous signal."""
    import numpy as np
    import pandas as pd
    from emg_classifier.features.uci_extractor import UCIFeatureExtractor

    config = yaml.safe_load(CONFIG_PATH.read_text())
    segment_dim, increment_dim = 10, 5
    config['segment_dim'] = segment_dim
    config['increment_dim'] = increment_dim
    config['feature_dim'] = 13
    config['channel_no'] = 1
    config['channel_subset'] = [1]
    # data.feature_dim is commented out in the shipped yaml (only set per-experiment
    # by scripts/run_experiment.py), but _process_chunk reads it directly.
    config.setdefault('data', {})['feature_dim'] = 13

    def make_session(user_id, session_idx, length):
        rng = np.random.default_rng(hash((user_id, session_idx)) % (2 ** 32))
        return pd.DataFrame({
            'channel1': rng.normal(size=length),
            'class': np.zeros(length, dtype=int),
            'user_id': user_id,
            'session_id': f"{user_id}_{session_idx}",
        })

    session_a = make_session('01', 1, 23)
    session_b = make_session('02', 1, 17)
    data = pd.concat([session_a, session_b], ignore_index=True)

    extractor = UCIFeatureExtractor(config)
    features, groups, session_ids, window_starts = extractor.extract_features(data, force_recompute=True)

    expected_a = len(range(0, len(session_a) - segment_dim, increment_dim))
    expected_b = len(range(0, len(session_b) - segment_dim, increment_dim))

    assert features.shape[0] == expected_a + expected_b
    assert set(groups) == {'01', '02'}
    assert list(groups).count('01') == expected_a
    assert list(groups).count('02') == expected_b
    assert set(session_ids) == {'01_1', '02_1'}
    assert set(window_starts) == set(range(0, 23 - segment_dim, increment_dim)) | \
                                  set(range(0, 17 - segment_dim, increment_dim))


def test_prepare_data_group_split_keeps_subjects_disjoint():
    """Regression test: BasePipeline._prepare_data, given subject groups, must never
    put the same subject's windows on both sides of the train/test split."""
    import numpy as np
    from emg_classifier.pipelines.base_pipeline import BasePipeline

    class DummyPipeline(BasePipeline):
        def setup(self):
            pass

        def _balance_data(self, X, y):
            return X, y  # no-op: isolate the split behavior from resampling

    n_subjects, windows_per_subject = 8, 15
    groups = np.repeat(np.arange(n_subjects), windows_per_subject).astype(float)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=len(groups))
    y = rng.integers(0, 3, size=len(groups))

    # Encode the subject id directly in a feature column so it can be recovered
    # (up to the fixed StandardScaler transform) after _prepare_data runs.
    features = np.column_stack([groups, noise, y])

    pipeline = DummyPipeline({'processing': {}})
    X_train, X_test, _, _ = pipeline._prepare_data(features, groups)

    train_subjects = set(np.round(X_train[:, 0], 6))
    test_subjects = set(np.round(X_test[:, 0], 6))

    assert train_subjects, "expected a non-empty training fold"
    assert test_subjects, "expected a non-empty test fold"
    assert train_subjects.isdisjoint(test_subjects)


def test_block_holdout_split_shares_subjects_but_keeps_sessions_whole():
    """Regression test: with split_strategy='block_holdout', the same subject should
    appear in both train and test (subject-dependent, matching the reference paper's
    own train/test-session protocol), but a single recording session's windows must
    never be split across train and test."""
    import numpy as np
    from emg_classifier.pipelines.base_pipeline import BasePipeline

    class DummyPipeline(BasePipeline):
        def setup(self):
            pass

        def _balance_data(self, X, y):
            return X, y  # no-op: isolate the split behavior from resampling

    n_subjects, sessions_per_subject, windows_per_session = 6, 2, 10
    groups, session_ids = [], []
    for subject in range(n_subjects):
        for session in range(sessions_per_subject):
            for _ in range(windows_per_session):
                groups.append(subject)
                session_ids.append(subject * 10 + session)  # globally unique per session
    groups = np.array(groups, dtype=float)
    session_ids = np.array(session_ids, dtype=float)

    rng = np.random.default_rng(0)
    noise = rng.normal(size=len(groups))
    y = rng.integers(0, 3, size=len(groups))

    # Encode subject id and session id in feature columns so both survive the fixed
    # per-column StandardScaler transform and can be recovered after the split.
    features = np.column_stack([groups, session_ids, noise, y])

    pipeline = DummyPipeline({'processing': {'split_strategy': 'block_holdout'}})
    X_train, X_test, _, _ = pipeline._prepare_data(features, groups, session_ids)

    train_subjects = set(np.round(X_train[:, 0], 6))
    test_subjects = set(np.round(X_test[:, 0], 6))
    train_sessions = set(np.round(X_train[:, 1], 6))
    test_sessions = set(np.round(X_test[:, 1], 6))

    assert train_subjects, "expected a non-empty training fold"
    assert test_subjects, "expected a non-empty test fold"
    assert train_subjects & test_subjects, "expected subjects to appear on both sides"
    assert train_sessions.isdisjoint(test_sessions), "a session's windows leaked across the split"


def test_intra_session_split_drops_straddling_windows():
    """Regression test: _intra_session_split must split a single session by time
    (train = early block, test = late block) and must never let a train window and a
    test window share any raw sample - i.e. any window whose [t, t+segment_dim) range
    straddles the cut point must be dropped from both sides, not assigned to either."""
    import numpy as np
    from emg_classifier.pipelines.base_pipeline import BasePipeline

    class DummyExtractor:
        segment_dim = 10

    class DummyPipeline(BasePipeline):
        def setup(self):
            pass

        def _balance_data(self, X, y):
            return X, y

    segment_dim, increment_dim, session_len = 10, 2, 100
    window_starts = np.arange(0, session_len - segment_dim, increment_dim)
    session_ids = np.array(['S1'] * len(window_starts))
    labels = np.zeros(len(window_starts), dtype=int)  # single class: degenerates to one global cut

    pipeline = DummyPipeline({'processing': {}})
    pipeline.feature_extractor = DummyExtractor()

    train_idx, test_idx = pipeline._intra_session_split(session_ids, window_starts, labels)

    assert len(train_idx) > 0
    assert len(test_idx) > 0
    assert len(train_idx) + len(test_idx) < len(window_starts), \
        "expected some straddling windows near the cut to be dropped"

    session_span = int(window_starts.max()) + segment_dim
    cut = int(session_span * 0.7)
    train_starts = window_starts[train_idx]
    test_starts = window_starts[test_idx]

    assert np.all(train_starts + segment_dim <= cut)
    assert np.all(test_starts >= cut)
    assert train_starts.max() < test_starts.min()


def test_intra_session_split_is_class_stratified():
    """Regression test: when one class finishes earlier in a session than another
    (as in the real UCI EMG data - see base_pipeline docstring, where e.g. class 1
    only occurs in the first ~56% of a recording), a single global time cut would
    starve the earlier-finishing class of any test examples entirely. The per-class
    cut must give it some test representation, and the safety pass must still
    guarantee zero raw-sample overlap between the two resulting sets even for windows
    right at the class-transition boundary."""
    import numpy as np
    from emg_classifier.pipelines.base_pipeline import BasePipeline

    class DummyExtractor:
        segment_dim = 10

    class DummyPipeline(BasePipeline):
        def setup(self):
            pass

        def _balance_data(self, X, y):
            return X, y

    segment_dim, increment_dim = 10, 2
    # One window per raster position (as in the real pipeline - each window has
    # exactly one mode-derived label). Class 0 finishes at t=48, class 1 takes over
    # from t=50 onward, so a single global cut (near t=68) would put ALL of class 0
    # in train.
    starts_0 = np.arange(0, 50, increment_dim)
    starts_1 = np.arange(50, 98, increment_dim)
    window_starts = np.concatenate([starts_0, starts_1])
    labels = np.concatenate([np.zeros(len(starts_0), dtype=int), np.ones(len(starts_1), dtype=int)])
    session_ids = np.array(['S1'] * len(window_starts))

    pipeline = DummyPipeline({'processing': {}})
    pipeline.feature_extractor = DummyExtractor()

    train_idx, test_idx = pipeline._intra_session_split(session_ids, window_starts, labels)

    test_labels = set(labels[test_idx])
    assert 0 in test_labels, "a global (non-stratified) cut would have starved class 0 of test examples"
    assert 1 in test_labels

    # Brute-force leakage check: no train window's raw range may overlap any test
    # window's raw range, regardless of class - including right at the class-0/
    # class-1 boundary, where windows from different classes genuinely overlap.
    for t_train in window_starts[train_idx]:
        for t_test in window_starts[test_idx]:
            assert not (t_train < t_test + segment_dim and t_test < t_train + segment_dim), \
                f"train window at {t_train} overlaps test window at {t_test}"
