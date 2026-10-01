"""Deterministic fixed-direction finite-difference HVP diagnostics.

Run on an already trained model. This does not train or change its weights.
Small relative discrepancies are necessary but not sufficient for validation;
this is not an independent exact-autograd Hessian or a converged eigenvalue.
"""
import math
import torch


def hvp_sensitivity(model, loss_fn, seed, epsilons=(0.001, 0.003, 0.01)):
    params = [p for p in model.parameters() if p.requires_grad]
    prior_training = model.training
    model.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    baseline = [p.detach().clone() for p in params]
    direction = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum(v.float().square().sum() for v in direction)).clamp_min(1e-12)
    direction = [v / norm.to(v.dtype) for v in direction]
    flat_direction = torch.cat([v.detach().float().cpu().flatten() for v in direction])

    def gradient(offset):
        with torch.no_grad():
            for p, base, v in zip(params, baseline, direction):
                p.copy_(base + offset * v)
        try:
            model.zero_grad(set_to_none=True)
            loss = loss_fn()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().flatten()
                              for p, g in zip(params, grads)])
        finally:
            with torch.no_grad():
                for p, base in zip(params, baseline):
                    p.copy_(base)
            model.zero_grad(set_to_none=True)

    probes = []
    previous = None
    try:
        g0, g0_repeat = gradient(0.0), gradient(0.0)
        repeat_relative_difference = float(torch.linalg.vector_norm(g0 - g0_repeat) / torch.linalg.vector_norm(g0).clamp_min(1e-12))
        del g0, g0_repeat
        for epsilon in epsilons:
            plus = gradient(epsilon)
            minus = gradient(-epsilon)
            hv = (plus - minus) / (2 * epsilon)
            del plus, minus
            probe = {"epsilon": epsilon, "rayleigh_fixed_direction": float(torch.dot(flat_direction, hv)),
                     "hvp_norm": float(torch.linalg.vector_norm(hv))}
            if previous is not None:
                probe["relative_difference_vs_previous_epsilon"] = float(torch.linalg.vector_norm(hv - previous) / torch.linalg.vector_norm(previous).clamp_min(1e-12))
                probe["cosine_vs_previous_epsilon"] = float(torch.dot(hv, previous) / (torch.linalg.vector_norm(hv) * torch.linalg.vector_norm(previous)).clamp_min(1e-12))
            if not all(math.isfinite(value) for value in probe.values()):
                raise RuntimeError("Nonfinite HVP diagnostic")
            probes.append(probe)
            print("HVP_SENSITIVITY", probe, flush=True)
            previous = hv
        restored = all(torch.equal(p.detach(), base) for p, base in zip(params, baseline))
        if not restored:
            raise RuntimeError("Curvature diagnostic failed to restore model weights")
        return {"scope": "numerical_diagnostic_not_eigenvalue_validation", "seed": seed,
                "dropout_disabled": True, "weights_exactly_restored": restored,
                "repeat_gradient_relative_difference": repeat_relative_difference, "probes": probes}
    finally:
        with torch.no_grad():
            for p, base in zip(params, baseline):
                p.copy_(base)
        model.zero_grad(set_to_none=True)
        model.train(prior_training)
