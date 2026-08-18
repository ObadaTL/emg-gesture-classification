from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

class BasePipeline(ABC):
    """Base class for EMG processing and classification pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = None
        self.feature_extractor = None
        self.model = None
        self.scaler = StandardScaler()
        
    @abstractmethod
    def setup(self):
        """Initialize data loader, feature extractor, and model"""
        pass
    
    @abstractmethod
    def _balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dataset-specific balancing strategy"""
        pass
    
    def run(self):
        """Execute the complete pipeline"""
        print("Starting pipeline execution...")
        
        # Load and preprocess data
        print("Step 1: Loading data...")
        data = self.data_loader.load_raw_data()
        processed_data = self.data_loader.preprocess_data(data)
        
        # Extract features
        print("Step 2: Extracting features...")
        features, groups, session_ids, window_starts = self.feature_extractor.extract_features(processed_data)

        # Split data
        print("Step 3: Splitting data...")
        X_train, X_test, y_train, y_test = self._prepare_data(features, groups, session_ids, window_starts)
        
        # Train model
        print("Step 4: Training model...")
        metrics = self.model.train(X_train, y_train, X_test, y_test)
        
        print("Pipeline execution completed.")
        return metrics
    
    def _block_holdout_split(self, groups: np.ndarray, session_ids: np.ndarray,
                              test_size: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Subject-dependent, leakage-free split: the same subjects appear in both
        train and test (matching the reference paper's own per-subject train/test
        session protocol), but held out by whole recording session rather than by
        individual window - so no overlapping sliding window is ever split across
        train and test. A subject with only one recording session contributes to
        train only (there's nothing to hold out for them).
        """
        rng = np.random.default_rng(random_state)
        train_mask = np.zeros(len(groups), dtype=bool)
        test_mask = np.zeros(len(groups), dtype=bool)

        for user in np.unique(groups):
            user_rows = np.where(groups == user)[0]
            user_sessions = np.unique(session_ids[user_rows])
            n_test_sessions = max(1, round(len(user_sessions) * test_size)) if len(user_sessions) > 1 else 0
            test_sessions = set(rng.choice(user_sessions, size=n_test_sessions, replace=False)) \
                if n_test_sessions else set()

            for row in user_rows:
                if session_ids[row] in test_sessions:
                    test_mask[row] = True
                else:
                    train_mask[row] = True

        return np.where(train_mask)[0], np.where(test_mask)[0]

    def _intra_session_split(self, session_ids: np.ndarray, window_starts: np.ndarray,
                              labels: np.ndarray, test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Same subject AND same physical sensor placement in both train and test:
        each recording session is split by time into an early train block and a late
        test block, holding the armband placement fixed and varying only time.

        The cut is computed PER CLASS within each session, not once globally: in this
        dataset gesture classes are presented in a rolling, progressively-later
        sequence through a recording (e.g. class 1 only occurs in the first ~56% of a
        session, class 6 only appears after ~46%), so a single global time cut starves
        whichever classes happen to finish early - the held-out test block ends up
        with zero examples of them. Cutting each class's own occurrence window at the
        same relative point keeps every class represented on both sides.

        Because different classes' time windows overlap, a per-class cut alone isn't
        enough to guarantee no shared raw samples between train and test (a window
        just past class A's cut could still overlap, in raw sample range, a window
        just before class B's cut). A final pass drops any pair of windows in the same
        session whose raw ranges overlap but landed on opposite sides.
        """
        segment_dim = self.feature_extractor.segment_dim
        assignment: Dict[int, str] = {}

        for session in np.unique(session_ids):
            session_rows = np.where(session_ids == session)[0]

            # Step 1: tentative assignment via each class's own time window.
            for cls in np.unique(labels[session_rows]):
                class_rows = session_rows[labels[session_rows] == cls]
                starts = window_starts[class_rows].astype(int)
                class_span = int(starts.max()) + segment_dim - int(starts.min())
                cut = int(starts.min()) + int(class_span * (1 - test_size))

                for row, t in zip(class_rows, starts):
                    if t + segment_dim <= cut:
                        assignment[row] = 'train'
                    elif t >= cut:
                        assignment[row] = 'test'
                    else:
                        assignment[row] = 'drop'  # straddles this class's own cut

            # Step 2: safety pass - drop any raw-overlapping pair assigned to
            # opposite sides, regardless of class. Windows in a session sit on a
            # fixed time raster, so only nearby neighbours (within segment_dim) can
            # possibly overlap; stop scanning ahead once starts move past that range.
            order = sorted(session_rows, key=lambda r: int(window_starts[r]))
            starts_sorted = [int(window_starts[r]) for r in order]
            for i in range(len(order)):
                t_i = starts_sorted[i]
                j = i + 1
                while j < len(order) and starts_sorted[j] < t_i + segment_dim:
                    r_i, r_j = order[i], order[j]
                    a_i, a_j = assignment.get(r_i), assignment.get(r_j)
                    if a_i in ('train', 'test') and a_j in ('train', 'test') and a_i != a_j:
                        assignment[r_i] = 'drop'
                        assignment[r_j] = 'drop'
                    j += 1

        train_idx = np.array([row for row, a in assignment.items() if a == 'train'])
        test_idx = np.array([row for row, a in assignment.items() if a == 'test'])
        n_dropped = len(assignment) - len(train_idx) - len(test_idx)
        if n_dropped:
            print(f"intra_session split: dropped {n_dropped} windows straddling a train/test time cut")

        return train_idx, test_idx

    def _prepare_data(self, features: np.ndarray, groups: Optional[np.ndarray] = None,
                       session_ids: Optional[np.ndarray] = None,
                       window_starts: Optional[np.ndarray] = None) -> tuple:
        """Prepare data for training.

        Splits BEFORE balancing so no synthetic/resampled point in the training fold
        is ever derived from a test-fold neighbor, and vice versa.

        `processing.split_strategy` in config selects the evaluation protocol:
          - "subject_independent" (default): GroupShuffleSplit on `groups` (subject
            ids) - every window belonging to a subject goes entirely to train or
            entirely to test. Tests generalization to a never-before-seen person.
          - "block_holdout": same subjects in both train and test, held out by whole
            recording session via `groups` + `session_ids` (see _block_holdout_split).
            In this dataset each subject's two sessions are their left/right arm, so
            this tests cross-arm transfer within the same person, not "same setup,
            later session" - it does NOT hold sensor placement fixed.
          - "intra_session": same session split by time, per class, via `session_ids`
            + `window_starts` (see _intra_session_split) - holds sensor placement
            fixed and tests generalization to later data from the exact same
            calibration. This is the closest match to the reference paper's own
            train/test-session protocol (repeated sessions with a stable setup).
        """
        print("Initial feature matrix shape:", features.shape)

        # Split features and labels
        X, y = features[:, :-1], features[:, -1]
        y = y.astype(int)

        print("Original class distribution:", np.bincount(y))

        split_strategy = self.config.get('processing', {}).get('split_strategy', 'subject_independent')

        # Split into train and test sets FIRST (before any resampling)
        if split_strategy == 'intra_session':
            if session_ids is None or window_starts is None:
                raise ValueError(
                    "split_strategy='intra_session' requires both session ids and window start positions."
                )
            print("Using intra_session split: same session split by time (per class), sensor placement held fixed.")
            train_idx, test_idx = self._intra_session_split(session_ids, window_starts, y)
        elif split_strategy == 'block_holdout':
            if groups is None or session_ids is None:
                raise ValueError(
                    "split_strategy='block_holdout' requires both subject groups and session ids."
                )
            print("Using block_holdout split: same subjects in train/test, held out by whole session.")
            train_idx, test_idx = self._block_holdout_split(groups, session_ids)
        elif groups is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y, groups))
        else:
            print("Warning: no subject groups available for this data - falling back to "
                  "a random stratified split. This is NOT subject-independent and may "
                  "leak overlapping windows from the same subject across train/test.")
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=0.3, random_state=42, stratify=y
            )

        X_train_raw, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test = y[train_idx], y[test_idx]

        print("Train class distribution (pre-balancing):", np.bincount(y_train_raw))
        print("Test class distribution (untouched):", np.bincount(y_test))

        # Apply dataset-specific balancing to the TRAINING fold only
        X_train, y_train = self._balance_data(X_train_raw, y_train_raw)

        print("Balanced train class distribution:", np.bincount(y_train))

        # Scale features (fit on train only)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Apply LDA for dimensionality reduction
        if self.config.get('processing', {}).get('use_lda', False):
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            n_components = min(self.config.get('processing', {}).get('lda_components', 4), 
                            len(np.unique(y_train)) - 1)  # LDA components cannot exceed classes-1
            print(f"\nApplying LDA dimensionality reduction with {n_components} components...")
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X_train = lda.fit_transform(X_train, y_train)
            X_test = lda.transform(X_test)
            print(f"Feature dimensions after LDA: {X_train.shape[1]}")
        
        # Print shapes after preprocessing
        print("\nAfter preprocessing:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_model(self, path: Path):
        """Save trained model"""
        self.model.save(path)
    
    def load_model(self, path: Path):
        """Load trained model"""
        self.model.load(path) 