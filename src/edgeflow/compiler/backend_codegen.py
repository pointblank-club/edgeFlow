"""Unified backend code generation interfaces and simple C backend.

This module defines a base interface for backend code generators and a
reference C backend that emits trivial C/H files from the lightweight IR
(`edgeflow_ir.IRGraph`). The intent is to demonstrate end-to-end plumbing and
provide a foundation for richer backends (LLVM, ARM CMSIS, TFLite flatbuffers).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


class BackendCodeGenerator(Protocol):
    """Abstract interface for backend code generators."""

    def generate(self, ir_graph: Any, target_config: Dict[str, Any]) -> List[str]:
        """Generate backend artifacts from IR.

        Args:
            ir_graph: The IR graph instance (edgeflow_ir.IRGraph).
            target_config: Target/device configuration and flags.

        Returns:
            List of file paths created.
        """


@dataclass
class CBackendOptions:
    output_dir: str = os.path.join("generated", "backend_c")
    header_name: str = "edge_model.h"
    source_name: str = "edge_model.c"


class EdgeBackendCCode:
    """Simple C backend code generator.

    Emits a header with node IDs and a source file with a skeletal run function
    that logs the execution order (as comments) based on the IR's topological
    sort. This is a minimal baseline to prove out the plumbing.
    """

    def __init__(self, options: CBackendOptions | None = None) -> None:
        self.options = options or CBackendOptions()

    def _sanitize_identifier(self, name: str) -> str:
        # Replace any character that isn't a letter, number, or underscore with '_'
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        # Task 3: Ensure it's a valid C token (cannot start with a digit)
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized

        return sanitized.upper()

    def generate(
        self, ir_graph: Any, target_config: Dict[str, Any]
    ) -> List[str]:  # noqa: D401
        # Allow overriding output_dir via target_config
        output_dir = target_config.get("output_dir", self.options.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Make sure we have an execution order
        try:
            if not getattr(ir_graph, "execution_order", []):
                ir_graph.topological_sort()
        except Exception:
            pass

        header_path = os.path.join(output_dir, self.options.header_name)
        source_path = os.path.join(output_dir, self.options.source_name)

        # Header content
        node_defs: List[str] = []
        for node_id, node in getattr(ir_graph, "nodes", {}).items():
            node_type = getattr(node, "op_type", None) or getattr(
                node, "node_type", None
            )
            node_type_str = (
                getattr(node_type, "value", str(node_type)) if node_type else "unknown"
            )
            node_defs.append(f"#define EF_NODE_{node_id.upper()} /* {node_type_str} */")

        header = "\n".join(
            [
                "#pragma once",
                "// EdgeFlow generated header - simple C backend",
                *node_defs,
                "void edge_model_init(void);",
                "void edge_model_run(void);",
                "",
            ]
        )

        # Source content
        exec_comments: List[str] = []
        for nid in getattr(ir_graph, "execution_order", []):
            node = ir_graph.nodes[nid]
            op = getattr(node, "op_type", None) or getattr(node, "node_type", None)
            op_str = getattr(op, "value", str(op)) if op else "unknown"
            exec_comments.append(f"// execute {nid} ({op_str})")

        source_lines = [
            "#include <stdio.h>",
            f'#include "{self.options.header_name}"',
            "",
            "void edge_model_init(void) {",
            "    // TODO: initialize weights/buffers if available",
            "}",
            "",
            "void edge_model_run(void) {",
            "    // Execution order derived from IR topological sort",
            *[f"    {line}" for line in exec_comments],
            "    (void)printf; // silence unused if printf removed",
            "}",
            "",
        ]
        source = "\n".join(source_lines)

        with open(header_path, "w", encoding="utf-8") as fh:
            fh.write(header)
        with open(source_path, "w", encoding="utf-8") as fs:
            fs.write(source)

        return [header_path, source_path]


def generate_backend_artifacts(
    ir_graph: Any, target_config: Dict[str, Any], target: str
) -> List[str]:
    """Dispatch to a backend code generator by target string.

    Args:
        ir_graph: The IR graph instance.
        target: Backend target identifier, e.g. "c", "llvm", "arm_cmsis".
        target_config: Backend configuration dictionary.

    Returns:
        List of generated file paths.
    """
    target_l = (target or "").strip().lower()
    if target_l in ("c", "c_backend", "edge_c"):
        return EdgeBackendCCode().generate(ir_graph, target_config)
    if target_l in ("rpi_c", "raspberry_pi", "arm_a"):
        return EdgeBackendRPIC().generate(ir_graph, target_config)
    if target_l in ("emulator_c", "ref_c", "portable_c"):
        return EdgeBackendEmulatorC().generate(ir_graph, target_config)
    raise ValueError(f"Unsupported backend target: {target}")


class EdgeBackendBase:
    """Shared helpers for C-like backends with lowering hooks."""

    def __init__(self, out_subdir: str) -> None:
        self.out_subdir = out_subdir

    def _ensure_dir(self, base_dir: str | None) -> str:
        base = base_dir or os.path.join("generated", self.out_subdir)
        os.makedirs(base, exist_ok=True)
        return base

    def _emit_header(self) -> str:
        return "\n".join(
            [
                "#pragma once",
                "#include <stdint.h>",
                "void edge_model_init(void);",
                "void edge_model_run(const float* input, float* output);",
                "",
            ]
        )

    def _emit_kernels(self) -> str:
        # Minimal portable kernels; RPi backend may override for NEON in future
        return "\n".join(
            [
                "static void relu_inplace(float* data, int n) {",
                "    for (int i = 0; i < n; ++i) if (data[i] < 0) data[i] = 0;",
                "}",
                "",
                "static void dense_naive(const float* x, const float* w, const float* b, int in, int out, float* y) {",
                "    for (int o = 0; o < out; ++o) {",
                "        float acc = b ? b[o] : 0.0f;",
                "        for (int i = 0; i < in; ++i) acc += x[i] * w[o*in + i];",
                "        y[o] = acc;",
                "    }",
                "}",
                "",
                "// Placeholder conv2d (expects NHWC, no padding/stride) for demonstration",
                "static void conv2d_naive(const float* x, const float* w, const float* b,",
                "    int n, int h, int w_, int c, int kh, int kw, int oc, float* y) {",
                "    (void)n; // batch 1 assumed",
                "    int oh = h - kh + 1;",
                "    int ow = w_ - kw + 1;",
                "    for (int oy = 0; oy < oh; ++oy) {",
                "      for (int ox = 0; ox < ow; ++ox) {",
                "        for (int o = 0; o < oc; ++o) {",
                "          float acc = b ? b[o] : 0.0f;",
                "          for (int ky = 0; ky < kh; ++ky) {",
                "            for (int kx = 0; kx < kw; ++kx) {",
                "              for (int ci = 0; ci < c; ++ci) {",
                "                int in_idx = ((oy+ky)*w_ + (ox+kx))*c + ci;",
                "                int k_idx = ((o*kh + ky)*kw + kx)*c + ci;",
                "                acc += x[in_idx] * w[k_idx];",
                "              }",
                "            }",
                "          }",
                "          int out_idx = ((oy*ow) + ox)*oc + o;",
                "          y[out_idx] = acc;",
                "        }",
                "      }",
                "    }",
                "}",
                "",
            ]
        )

    def _infer_buffer_sizes(self, ir_graph: Any) -> tuple[int, int]:
        # Very rough inference: input/output dims from node properties if present
        def _parse_shape(val: Any) -> int:
            if isinstance(val, str):
                dims = [int(x) if x.isdigit() else 1 for x in val.split(",")]
                prod = 1
                for d in dims:
                    prod *= d
                return prod
            if isinstance(val, (list, tuple)):
                prod = 1
                for d in val:
                    prod *= int(d) if isinstance(d, int) else 1
                return prod
            return 1

        # Unified IR uses tensors; try first
        if (
            hasattr(ir_graph, "tensors")
            and isinstance(getattr(ir_graph, "tensors"), dict)
            and ir_graph.tensors
        ):
            # Look for obvious names
            in_tensor = ir_graph.tensors.get("input") or next(
                iter(ir_graph.tensors.values())
            )
            out_tensor = ir_graph.tensors.get("output") or next(
                reversed(ir_graph.tensors.values())
            )
            in_elems = _parse_shape(getattr(in_tensor, "shape", getattr(in_tensor, "dimensions", [1])))  # type: ignore[arg-type]
            out_elems = _parse_shape(getattr(out_tensor, "shape", getattr(out_tensor, "dimensions", [1])))  # type: ignore[arg-type]
            return in_elems, out_elems

        input_node = (
            ir_graph.nodes.get("input_0") if hasattr(ir_graph, "nodes") else None
        )
        output_node = (
            ir_graph.nodes.get("output_0") if hasattr(ir_graph, "nodes") else None
        )
        in_elems = (
            _parse_shape(
                getattr(input_node, "properties", {}).get("input_shape", "1,224,224,3")
            )
            if input_node
            else 1
        )
        out_elems = (
            _parse_shape(
                getattr(output_node, "properties", {}).get("output_shape", [1, 1000])
            )
            if output_node
            else 1
        )
        return in_elems, out_elems

    def _emit_model(
        self, ir_graph: Any, target_config: Dict[str, Any], header_name: str
    ) -> str:
        in_elems, out_elems = self._infer_buffer_sizes(ir_graph)
        body = [
            f'#include "{header_name}"',
            self._emit_kernels(),
            "",
            "void edge_model_init(void) {",
            "    // TODO: allocate/load weights if available",
            "}",
            "",
            "void edge_model_run(const float* input, float* output) {",
            f"    // buffers sized from IR: in={in_elems}, out={out_elems}",
        ]

        # Walk in execution order and emit placeholders per op
        try:
            order = ir_graph.execution_order or ir_graph.topological_sort()
        except Exception:
            order = list(ir_graph.nodes.keys())

        # Try unified IR first
        def _emit_from_uir():
            if not hasattr(ir_graph, "nodes"):
                return False
            # Unified IR nodes carry operation_type.value
            try:
                for nid, node in ir_graph.nodes.items():
                    op_val = getattr(getattr(node, "operation_type", None), "value", "")
                    if op_val == "conv2d":
                        body.append(
                            "    // conv2d_naive(input, Wconv, Bconv, N,H,W,C, Kh, Kw, Oc, out)"
                        )
                    elif op_val == "dense":
                        body.append("    // dense_naive(x, Wdense, Bdense, in, out, y)")
                    elif op_val == "relu":
                        body.append("    // relu_inplace(buffer, length)")
                return True
            except Exception:
                return False

        if not _emit_from_uir():
            for nid in order:
                node = ir_graph.nodes[nid]
                op = getattr(node, "op_type", None) or getattr(node, "node_type", None)
                op_str = getattr(op, "value", str(op)) if op else "unknown"
                op_l = op_str.lower() if isinstance(op_str, str) else str(op_str)
                if "conv2d" in op_l or op_l == "conv2d":
                    body.append("    // conv2d_naive(...)  // TODO: wire real params")
                elif "dense" in op_l or op_l == "dense":
                    body.append("    // dense_naive(...)   // TODO: wire real params")
                elif "relu" in op_l:
                    body.append("    // relu_inplace(...)  // TODO: length")
                else:
                    body.append(f"    // pass-through {nid} ({op_str})")

        body.extend(
            [
                "}",
                "",
            ]
        )
        return "\n".join(body)


class EdgeBackendRPIC(EdgeBackendBase):
    """Raspberry Pi oriented C backend (ARMv7/ARMv8)."""

    def __init__(self) -> None:
        super().__init__(out_subdir="backend_rpi_c")

    def generate(self, ir_graph: Any, target_config: Dict[str, Any]) -> List[str]:
        out_dir = self._ensure_dir(target_config.get("output_dir"))
        header_name = "edge_model.h"
        src_name = "edge_model.c"
        main_name = "main.c"
        mk_name = "Makefile"

        header_path = os.path.join(out_dir, header_name)
        src_path = os.path.join(out_dir, src_name)
        main_path = os.path.join(out_dir, main_name)

        with open(header_path, "w", encoding="utf-8") as fh:
            fh.write(self._emit_header())
        with open(src_path, "w", encoding="utf-8") as fs:
            fs.write(self._emit_model(ir_graph, target_config, header_name))
        with open(main_path, "w", encoding="utf-8") as fm:
            fm.write(
                "\n".join(
                    [
                        f'#include "{header_name}"',
                        "#include <stdio.h>",
                        "#include <stdlib.h>",
                        "",
                        "int main() {",
                        "    // Simple demo main: allocates input/output and runs model",
                        "    size_t in_sz = 1*224*224*3; // default, replace when wiring exact dims",
                        "    size_t out_sz = 1000;",
                        "    float* in = (float*)calloc(in_sz, sizeof(float));",
                        "    float* out = (float*)calloc(out_sz, sizeof(float));",
                        "    edge_model_init();",
                        "    edge_model_run(in, out);",
                        '    printf("done\\n");',
                        "    free(in); free(out);",
                        "    return 0;",
                        "}",
                        "",
                    ]
                )
            )

        mk_path = os.path.join(out_dir, mk_name)
        with open(mk_path, "w", encoding="utf-8") as mk:
            mk.write(
                "\n".join(
                    [
                        "CC ?= gcc",
                        "CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter",
                        "LDFLAGS ?=",
                        "SRC := edge_model.c main.c",
                        "OBJ := $(SRC:.c=.o)",
                        "TARGET := edge_model_demo",
                        "all: $(TARGET)",
                        "$(TARGET): $(OBJ)",
                        "\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)",
                        "%.o: %.c edge_model.h",
                        "\t$(CC) $(CFLAGS) -c -o $@ $<",
                        ".PHONY: clean",
                        "clean:",
                        "\trm -f $(OBJ) $(TARGET)",
                        "",
                    ]
                )
            )

        return [header_path, src_path, main_path, mk_path]


class EdgeBackendEmulatorC(EdgeBackendBase):
    """Portable reference-kernel C backend for emulation."""

    def __init__(self) -> None:
        super().__init__(out_subdir="backend_emulator_c")

    def generate(self, ir_graph: Any, target_config: Dict[str, Any]) -> List[str]:
        out_dir = self._ensure_dir(target_config.get("output_dir"))
        header_name = "edge_model.h"
        src_name = "edge_model.c"
        header_path = os.path.join(out_dir, header_name)
        src_path = os.path.join(out_dir, src_name)
        with open(header_path, "w", encoding="utf-8") as fh:
            fh.write(self._emit_header())
        with open(src_path, "w", encoding="utf-8") as fs:
            fs.write(self._emit_model(ir_graph, target_config, header_name))
        return [header_path, src_path]
