"""Avoid host policy copies when Miles keeps the sole actor resident on GPU."""


def install(actor_type):
    # Ray copies methods into a generated subclass before this init hook runs.
    # Patch those already-created wrappers as well as the original class.
    for subclass in actor_type.__subclasses__():
        install(subclass)
    if actor_type.__dict__.get("_archlab_resident_policy_installed", False):
        return
    original_enabled = actor_type._enable_weight_backup
    original_weights = actor_type._get_actor_weights

    def resident(actor):
        return (actor.args.colocate and not actor.args.offload_train
                and not actor.with_ref and not actor.with_opd_teacher
                and not actor.args.keep_old_actor)

    def enabled(actor):
        return False if resident(actor) else original_enabled.fget(actor)

    def weights(actor):
        if not resident(actor):
            return original_weights(actor)
        if actor._active_model_tag != "actor":
            raise RuntimeError("resident policy sync requires the live actor")
        return dict(actor._named_actor_weights())

    actor_type._enable_weight_backup = property(enabled)
    actor_type._get_actor_weights = weights
    actor_type._archlab_resident_policy_installed = True
