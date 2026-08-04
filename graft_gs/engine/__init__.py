"""Staged distributed training, synchronization, losses, and checkpointing."""

from .trainer import (
    AtlasDDPSynchronizer,
    assert_local_cuda_allocator_ownership,
    bind_local_cuda_device,
    GraftGSTrainer,
    TrainerConfig,
    TrainingPhase,
)
from .supervision import SurfaceTargetConfig, derive_feasible_surface_target
from .checkpoints import (
    CheckpointLoadReport,
    MODEL_EXECUTION_POLICY_PATHS,
    load_graft_checkpoint,
    model_config_differences,
    prepare_model_for_checkpoint,
    validate_model_config_compatibility,
    validate_precision_policy,
    validate_trellis_prior_policy,
)
from .configuration import (
    load_loss_weights,
    load_precision_policy,
    load_progress_config,
    load_server_config,
    load_training_profiler_config,
    load_trellis_prior_config,
)
from .precision import NativePrecisionPolicy
from .teacher_refinement import (
    TeacherBundleConfig,
    TeacherBundleResult,
    LoadedTeacherBundle,
    TopologyFixedTeacherBundleRefiner,
    load_teacher_bundle,
)

__all__ = [
    "AtlasDDPSynchronizer",
    "assert_local_cuda_allocator_ownership",
    "bind_local_cuda_device",
    "CheckpointLoadReport",
    "MODEL_EXECUTION_POLICY_PATHS",
    "GraftGSTrainer",
    "SurfaceTargetConfig",
    "TrainerConfig",
    "TrainingPhase",
    "TeacherBundleConfig",
    "TeacherBundleResult",
    "LoadedTeacherBundle",
    "NativePrecisionPolicy",
    "TopologyFixedTeacherBundleRefiner",
    "derive_feasible_surface_target",
    "load_graft_checkpoint",
    "load_loss_weights",
    "load_precision_policy",
    "load_progress_config",
    "load_training_profiler_config",
    "load_teacher_bundle",
    "load_server_config",
    "load_trellis_prior_config",
    "model_config_differences",
    "prepare_model_for_checkpoint",
    "validate_model_config_compatibility",
    "validate_precision_policy",
    "validate_trellis_prior_policy",
]
