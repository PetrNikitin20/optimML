"""Autograd Hessian diagnostics for the last MLP down-projection LoRA-B.

This is a restricted parameter subspace, NOT the full LoRA/model Hessian.
No weights are changed. Frozen upstream layers and disabled embedding-gradient
hooks avoid keeping a full-model second-order graph on a T4.
"""
import re
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def projected_hessian_exact(model, loss_fn, seed, iterations=8):
    all_params = list(model.named_parameters())
    candidates = [(name, p) for name, p in all_params
                  if p.requires_grad and ".mlp.down_proj.lora_B." in name and name.endswith(".weight")]
    if not candidates:
        raise RuntimeError("No MLP down-projection LoRA-B found")
    target_name, target = max(candidates, key=lambda item: int(re.search(r"\.layers\.(\d+)\.", item[0]).group(1)))
    flags = [(p, p.requires_grad) for _, p in all_params]
    previous_mode = model.training
    model.eval()
    hook_disabled = False
    try:
        model.disable_input_require_grads()
        hook_disabled = True
        for p, _ in flags:
            p.requires_grad_(False)
        target.requires_grad_(True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        vector = torch.randn_like(target)
        vector = vector / vector.norm().clamp_min(1e-12)

        def hvp(v):
            model.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.MATH):
                loss = loss_fn()
                gradient, = torch.autograd.grad(loss, target, create_graph=True)
                product = (gradient * v).sum()
                hv, = torch.autograd.grad(product, target)
            return hv.detach().float()

        # Determinism and symmetry checks before interpreting any spectrum.
        first = hvp(vector)
        repeated = hvp(vector)
        repeat_error = float((first - repeated).norm() / first.norm().clamp_min(1e-12))
        other = torch.randn_like(target)
        other = other / other.norm().clamp_min(1e-12)
        h_other = hvp(other)
        left, right = (other * first).sum(), (vector * h_other).sum()
        symmetry_absolute_error = float((left - right).abs())
        symmetry_relative_error = float((left - right).abs() / torch.maximum(left.abs(), right.abs()).clamp_min(1e-12))
        history = []
        for iteration in range(iterations):
            product = hvp(vector)
            rayleigh = float((vector * product).sum())
            product_norm = product.norm()
            if not bool(torch.isfinite(product_norm)) or float(product_norm) == 0:
                raise RuntimeError("Nonfinite or zero projected Hessian-vector product")
            residual = float((product - rayleigh * vector).norm() / product_norm)
            history.append({"iteration": iteration + 1, "rayleigh": rayleigh, "relative_residual": residual})
            print("PROJECTED_HESSIAN", history[-1], flush=True)
            vector = (product / product_norm).to(target.dtype)
        return {"scope": "last_MLP_down_projection_LoRA_B_only_not_full_model_Hessian",
                "parameter_name": target_name, "n_parameters": target.numel(),
                "method": "exact_autograd_double_backward_math_SDPA",
                "interpretation": "dominant_magnitude_Rayleigh_quotient_not_largest_algebraic_eigenvalue",
                "dropout_disabled": True, "weights_modified": False,
                "repeat_hvp_relative_difference": repeat_error,
                "symmetry_absolute_error": symmetry_absolute_error,
                "symmetry_relative_error": symmetry_relative_error,
                "seed": seed, "iterations": iterations, "history": history,
                "rayleigh": history[-1]["rayleigh"], "relative_residual": history[-1]["relative_residual"]}
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        if hook_disabled:
            model.enable_input_require_grads()
        model.zero_grad(set_to_none=True)
        model.train(previous_mode)
