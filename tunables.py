"""Generic dataclass <-> argparse bridge.

Any dataclass field marked with ``field(metadata={"cli": True})`` becomes a
``--flag-name`` argument whose type and default come straight from the field
itself. This is what lets the CLI (``cli.py``) and the GUI (``gui_app.py``)
stay in sync with the actual code without ever hand-duplicating a default:
change a dataclass field's default and every surface that reads it (argparse
``--help``, the GUI's pre-filled form) picks it up automatically.

Fields with no ``cli`` metadata (or ``cli: False``) are left alone -- this is
how internal/runtime-state fields (``pyboy``, ``in_battle_ticks``, ...) stay
out of the CLI/GUI surface without a separately maintained blocklist.
"""
from __future__ import annotations

import argparse
import dataclasses
import typing


def _flag_name(field_name: str, flag_prefix: str) -> str:
    return "--" + flag_prefix + field_name.replace("_", "-")


def _dest_name(field_name: str, flag_prefix: str) -> str:
    return (flag_prefix + field_name).replace("-", "_")


def _field_default(f: "dataclasses.Field") -> object:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()
    raise ValueError(
        f"cli=True field {f.name!r} has no default -- every tunable field "
        f"needs one so its CLI/GUI default matches the code's current value"
    )


def cli_fields(dataclass_type: type) -> list["dataclasses.Field"]:
    """Every field of ``dataclass_type`` marked ``metadata={'cli': True}``."""
    return [
        f for f in dataclasses.fields(dataclass_type) if f.metadata.get("cli")
    ]


def add_dataclass_args(
    parser: argparse.ArgumentParser,
    dataclass_type: type,
    *,
    group_title: str | None = None,
    flag_prefix: str = "",
) -> None:
    """Add one ``--flag`` per ``cli=True`` field of ``dataclass_type``."""
    target = parser.add_argument_group(group_title) if group_title else parser
    hints = typing.get_type_hints(dataclass_type)

    for f in cli_fields(dataclass_type):
        flag = _flag_name(f.name, flag_prefix)
        dest = _dest_name(f.name, flag_prefix)
        default = _field_default(f)
        help_text = f.metadata.get("help", "")
        py_type = hints.get(f.name, type(default))

        if py_type is bool:
            target.add_argument(
                flag,
                dest=dest,
                action=argparse.BooleanOptionalAction,
                default=default,
                help=f"{help_text} (default: {default})".strip(),
            )
        else:
            target.add_argument(
                flag,
                dest=dest,
                type=py_type,
                default=default,
                metavar=py_type.__name__.upper(),
                help=f"{help_text} (default: {default})".strip(),
            )


def kwargs_from_args(
    args: argparse.Namespace,
    dataclass_type: type,
    *,
    flag_prefix: str = "",
) -> dict[str, object]:
    """Inverse of :func:`add_dataclass_args`: pull the parsed values for
    ``dataclass_type``'s ``cli=True`` fields back out of ``args`` into a
    plain kwargs dict, keyed by the dataclass's own field names -- ready to
    splat straight into its constructor."""
    out: dict[str, object] = {}
    for f in cli_fields(dataclass_type):
        dest = _dest_name(f.name, flag_prefix)
        if hasattr(args, dest):
            out[f.name] = getattr(args, dest)
    return out
