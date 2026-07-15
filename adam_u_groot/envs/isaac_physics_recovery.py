"""Work around PhysX tensor views invalidated while URDF meshes load at env startup."""

from __future__ import annotations


def install_manager_env_physics_recovery_patch() -> None:
    """Rebind PhysX views after sim.reset() when URDF import invalidates them."""
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    if getattr(ManagerBasedEnv, "_adam_u_physics_recovery_patch", False):
        return

    original_init_sim = ManagerBasedEnv._init_sim

    def _init_sim_with_startup_fix(self):
        sim = self.sim
        original_reset = sim.reset

        def reset_with_rebind(*args, **kwargs):
            from isaaclab.sim.utils.stage import use_stage

            with use_stage(sim.stage):
                original_reset(*args, **kwargs)

            print("[INFO] Rebinding PhysX views after URDF/mesh import...", flush=True)
            _rebind_physics_views(self.scene)

        sim.reset = reset_with_rebind
        try:
            return original_init_sim(self)
        finally:
            sim.reset = original_reset

    ManagerBasedEnv._init_sim = _init_sim_with_startup_fix
    ManagerBasedEnv._adam_u_physics_recovery_patch = True


def _rebind_physics_views(scene) -> None:
    """Create fresh simulation views; physics is already running so skip stage attach/warmup."""
    import omni.physics.tensors
    from isaaclab.physics import PhysicsEvent
    from isaaclab.sim.utils.stage import get_current_stage_id
    from isaaclab_physx.physics.physx_manager import IsaacEvents, PhysxManager

    PhysxManager._view = None
    PhysxManager._view_warp = None
    PhysxManager._view_created = False

    for group in (scene._articulations, scene._rigid_objects, scene._rigid_object_collections):
        for asset in group.values():
            asset._is_initialized = False
            if hasattr(asset, "_root_view"):
                asset._root_view = None
            if hasattr(asset, "_data") and hasattr(asset._data, "_root_view"):
                asset._data._root_view = None

    for sensor in scene._sensors.values():
        sensor._is_initialized = False

    stage_id = get_current_stage_id()
    print("[INFO] Creating fresh PhysX simulation views...", flush=True)
    PhysxManager._view = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
    PhysxManager._view_warp = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
    print("[INFO] Fresh PhysX simulation views created.", flush=True)
    if PhysxManager._view:
        PhysxManager._view.set_subspace_roots("/")
    if PhysxManager._view_warp:
        PhysxManager._view_warp.set_subspace_roots("/")
    PhysxManager._view_created = True

    dt = PhysxManager.get_physics_dt()
    print("[INFO] Synchronizing fresh PhysX views...", flush=True)
    PhysxManager._physx.update_simulation(dt, 0.0)
    if PhysxManager._physx_sim is not None:
        PhysxManager._physx_sim.fetch_results()
    print("[INFO] Fresh PhysX views synchronized.", flush=True)

    PhysxManager._event_bus.dispatch_event(IsaacEvents.SIMULATION_VIEW_CREATED.value, payload={})
    PhysxManager.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})
    PhysxManager._event_bus.dispatch_event(IsaacEvents.PHYSICS_READY.value, payload={})
    print("[INFO] Fresh PhysX views dispatched.", flush=True)
