"""AST-based config injection into train.py."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigInjector:
    """Inject HPO-suggested values into train.py by modifying AST assignments."""

    def __init__(self, train_py_path: Path):
        self.path = train_py_path

    def inject(self, config: dict[str, object]) -> str:
        """Inject config values into train.py and return modified source.

        Uses AST parsing to find ALL_CAPS = literal assignments and replaces
        them with HPO-suggested values. Falls back to regex for edge cases.

        Args:
            config: Mapping of HP name → value to inject.

        Returns:
            Modified train.py source code.
        """
        source = self.path.read_text()
        remaining = dict(config)

        # Try AST-based injection first
        source, remaining = self._inject_ast(source, remaining)

        # Regex fallback for anything the AST pass missed
        if remaining:
            source, remaining = self._inject_regex(source, remaining)

        if remaining:
            logger.warning("Could not inject these HPs (not found in train.py): %s", list(remaining.keys()))

        return source

    def inject_and_write(self, config: dict[str, object]) -> None:
        """Inject config values and write back to train.py."""
        modified = self.inject(config)
        self.path.write_text(modified)
        logger.info("Wrote modified train.py with %d HP values", len(config))

    def _inject_ast(self, source: str, config: dict[str, object]) -> tuple[str, dict[str, object]]:
        """AST-based injection: parse, find assignments, replace values."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("Failed to parse train.py with AST, falling back to regex")
            return source, config

        lines = source.splitlines(keepends=True)
        remaining = dict(config)
        # Process replacements in reverse line order to preserve line numbers
        replacements: list[tuple[int, str, str, object]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name not in remaining:
                continue

            new_value = remaining.pop(name)
            replacements.append((node.lineno, name, self._format_value(new_value), new_value))

        # Apply in reverse order to preserve line numbers
        for lineno, name, formatted, raw_value in sorted(replacements, key=lambda x: x[0], reverse=True):
            idx = lineno - 1
            old_line = lines[idx]
            # Replace the assignment value while preserving comments
            new_line = re.sub(
                rf"^(\s*{re.escape(name)}\s*=\s*)(.+?)(\s*#.*)?$",
                rf"\g<1>{formatted}\g<3>",
                old_line,
            )
            if new_line == old_line:
                # Simple fallback: just replace the whole line
                indent = len(old_line) - len(old_line.lstrip())
                new_line = " " * indent + f"{name} = {formatted}\n"
            lines[idx] = new_line
            logger.debug("Injected %s = %s (line %d)", name, formatted, lineno)

        return "".join(lines), remaining

    def _inject_regex(self, source: str, config: dict[str, object]) -> tuple[str, dict[str, object]]:
        """Regex fallback for HPs not found by AST pass."""
        remaining = dict(config)
        for name, value in list(remaining.items()):
            formatted = self._format_value(value)
            pattern = rf"^(\s*{re.escape(name)}\s*=\s*)(.+?)(\s*#.*)?$"
            new_source, count = re.subn(pattern, rf"\g<1>{formatted}\g<3>", source, flags=re.MULTILINE)
            if count > 0:
                source = new_source
                remaining.pop(name)
                logger.debug("Injected %s = %s via regex", name, formatted)

        return source, remaining

    @staticmethod
    def _format_value(value: object) -> str:
        """Format a Python value for source code insertion."""
        import numpy as np
        # Convert numpy types to native Python first
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, (np.str_, np.bytes_)):
            value = str(value)

        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            # Use scientific notation for very small/large values
            if 0 < abs(value) < 0.001 or abs(value) > 100000:
                return f"{value:.6e}"
            return repr(value)
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, (list, tuple)):
            return repr(value)
        return repr(value)
