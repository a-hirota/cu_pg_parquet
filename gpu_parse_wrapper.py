"""
Alias for backward compatibility: allow `import gpu_parse_wrapper` at top level.
"""
from gpupaser.gpu_parse_wrapper import parse_binary_chunk_gpu

__all__ = ["parse_binary_chunk_gpu"]
