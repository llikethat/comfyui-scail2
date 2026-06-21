# Intentionally empty. See scail2_wan/__init__.py for rationale.
#
# Upstream imports WanModel and SCAILModel here, which require `diffusers`.
# Our pipeline uses SCAIL2Model from model_scail2.py, which has no diffusers
# dependency. Removing the eager imports lets users run the SCAIL-2 path
# without installing diffusers (still required for upstream's legacy paths,
# but never invoked by this wrapper).
