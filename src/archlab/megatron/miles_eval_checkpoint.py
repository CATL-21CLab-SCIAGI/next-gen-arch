"""Reject training-state resume from explicit model-only native checkpoints."""

from pathlib import Path


def check_eval_load(args, load_dir=None):
    selected_load = load_dir or getattr(args, "load", None)
    if selected_load is None:
        return
    root = Path(selected_load)
    selected = root
    if not (root / "EVAL_ONLY.json").exists():
        step = getattr(args, "ckpt_step", None)
        # This pinned native loader treats ckpt_step=0 as unset.
        if not step:
            tracker = root / "latest_checkpointed_iteration.txt"
            if not tracker.exists():
                return
            step = tracker.read_text().strip()
        selected = root / ("release" if step == "release" else f"iter_{int(step):07d}")
    if not (selected / "EVAL_ONLY.json").exists():
        return
    if (not getattr(args, "no_load_optim", False)
            or not getattr(args, "no_load_rng", False)
            or getattr(args, "load_main_params_from_ckpt", False)):
        raise ValueError(f"Evaluation-only checkpoint cannot resume training state: {selected}; "
                         "require --no-load-optim --no-load-rng and no --load-main-params-from-ckpt")
