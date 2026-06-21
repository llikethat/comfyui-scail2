# Intentionally empty.
#
# The upstream zai-org/SCAIL-2 repo re-exports the full public surface here
# (WanVAE, SCAIL2Pipeline, T5*, etc.). For our ComfyUI wrapper this eager
# import chain pulls in diffusers, decord, opencv and others even when the
# user only loads one component. We import the specific submodules we need
# from the node code instead, so missing optional deps surface as targeted
# errors on the relevant node, not as a blanket package-init failure.
