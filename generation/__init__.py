"""
Unconditional DDPM Pipeline — self-contained sub-package.

Generates three comparison images from the SAME initial noise:
  1. Baseline   — clean UNet, no patching
  2. Patched    — direct h-space patching (v added to mid_block)
  3. CFG        — dual-pass: unpatched vs patched, combined via CFG scale
"""
