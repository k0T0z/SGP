---
name: shader-gen-read-guard
description: Block reading the shader-gen submodule unless the user explicitly asks. Use before any exploration that could enter shader-gen, including grep, glob, or file reads.
---

# Shader-Gen Read Guard

The `shader-gen/` submodule is off-limits for reading unless the user explicitly tells you to read it.

## Rules

- Do not read, grep, glob, or list files under `shader-gen/`.
- Do not follow imports or includes into `shader-gen/` to "understand" a type, field, or range.
- Keep searches scoped to the superproject (e.g. `sga/`, `src/`, `renderer.py`) so results never resolve inside `shader-gen/`.
- Treat `shader-gen` internals as a black box: rely on the Python bindings, the encoded graph string format, and existing project memory instead of reading the submodule source.

## Allowed only when explicitly told

If the user says to read, open, inspect, or modify `shader-gen` (or names a specific file inside it), proceed normally for that request. Do not generalize that permission to later tasks.

## If a task seems to require it

Stop and ask the user whether you may read `shader-gen`. Do not read it speculatively to answer the question.
