---
name: Always verify paths precisely
description: Never confuse Flash-Lite and Pro Preview paths — check exact directory names before reporting
type: feedback
---

Never confuse model variant paths (e.g., Flash-Lite vs Pro Preview). Always check the exact directory name before reporting results.

**Why:** Misidentifying gemini31_benchmark (Flash-Lite) as Pro Preview is a serious error that could lead to wrong numbers in the paper.

**How to apply:** When looking up Gemini results, always verify the full path contains the correct model name (e.g., `gemini_3_1_pro_preview` vs `gemini_3_1_flash_lite_preview`). Cross-check with the memory file `project_gemini_pro_paths.md` which documents the split across directories.
