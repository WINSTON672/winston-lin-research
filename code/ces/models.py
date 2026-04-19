"""
models.py
---------
Known inference FLOPs and benchmark scores for major AI models.
FLOPs are per-token inference estimates (normalized to 1T tokens = 1.0).

Sources:
- Chinchilla scaling laws (Hoffmann et al., 2022)
- Model cards and technical reports
- MLPerf inference benchmarks
"""

# Inference FLOPs per token (in teraFLOPs, normalized)
# Estimated from: 2 * parameters * 1 forward pass
MODEL_FLOPS = {
    "gpt-4":          1800.0,   # ~1.8T params estimated (MoE)
    "gpt-4o":          600.0,   # optimized, ~600B effective
    "gpt-3.5-turbo":   175.0,
    "claude-3-opus":   500.0,   # estimated
    "claude-3-sonnet": 200.0,
    "claude-3-haiku":   20.0,
    "gemini-ultra":   1000.0,
    "gemini-pro":      340.0,
    "llama-3-70b":      70.0,
    "llama-3-8b":        8.0,
    "mistral-7b":        7.0,
    "mixtral-8x7b":     46.7,   # MoE: 8 experts, 7B each, 2 active
    "phi-3-mini":        3.8,
    "human":          1000.0,   # ~1 petaFLOP (brain estimate, Moravec 1998)
}

# MMLU benchmark scores (0-1 accuracy across 57 subjects)
# Source: published papers and leaderboards
MODEL_MMLU = {
    "gpt-4":           0.864,
    "gpt-4o":          0.887,
    "gpt-3.5-turbo":   0.700,
    "claude-3-opus":   0.868,
    "claude-3-sonnet": 0.790,
    "claude-3-haiku":  0.752,
    "gemini-ultra":    0.904,
    "gemini-pro":      0.791,
    "llama-3-70b":     0.820,
    "llama-3-8b":      0.684,
    "mistral-7b":      0.641,
    "mixtral-8x7b":    0.706,
    "phi-3-mini":      0.688,
    "human":           0.898,   # estimated expert human performance
}

# HumanEval benchmark scores (code generation, pass@1)
MODEL_HUMANEVAL = {
    "gpt-4":           0.870,
    "gpt-4o":          0.902,
    "gpt-3.5-turbo":   0.481,
    "claude-3-opus":   0.849,
    "claude-3-sonnet": 0.730,
    "claude-3-haiku":  0.750,
    "gemini-ultra":    0.745,
    "gemini-pro":      0.677,
    "llama-3-70b":     0.814,
    "llama-3-8b":      0.620,
    "mistral-7b":      0.264,
    "mixtral-8x7b":    0.406,
    "phi-3-mini":      0.582,
    "human":           1.000,
}


def list_models() -> list:
    return [m for m in MODEL_FLOPS if m != "human"]
