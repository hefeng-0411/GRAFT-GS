"""Staged distributed training, synchronization, losses, and checkpointing."""

from .trainer import (
    AtlasDDPSynchronizer,
    GraftGSTrainer,
    TrainerConfig,
    TrainingPhase,
)
from .supervision import SurfaceTargetConfig, derive_feasible_surface_target
from .checkpoints import (
    CheckpointLoadReport,
    load_graft_checkpoint,
    prepare_model_for_checkpoint,
    validate_trellis_prior_policy,
)
from .configuration import load_loss_weights, load_server_config, load_trellis_prior_config
from .teacher_refinement import (
    TeacherBundleConfig,
    TeacherBundleResult,
    LoadedTeacherBundle,
    TopologyFixedTeacherBundleRefiner,
    load_teacher_bundle,
)

__all__ = [
    "AtlasDDPSynchronizer",
    "CheckpointLoadReport",
    "GraftGSTrainer",
    "SurfaceTargetConfig",
    "TrainerConfig",
    "TrainingPhase",
    "TeacherBundleConfig",
    "TeacherBundleResult",
    "LoadedTeacherBundle",
    "TopologyFixedTeacherBundleRefiner",
    "derive_feasible_surface_target",
    "load_graft_checkpoint",
    "load_loss_weights",
    "load_teacher_bundle",
    "load_server_config",
    "load_trellis_prior_config",
    "prepare_model_for_checkpoint",
    "validate_trellis_prior_policy",
]
