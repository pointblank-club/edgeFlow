"""Extended tests for parser module to achieve 90% coverage."""

import os
import tempfile
from edgeflow.parser import (
    EdgeFlowParserError,
    parse_edgeflow_file,
    parse_edgeflow_string,
    validate_config,
)
from edgeflow.compiler.backend_codegen import EdgeBackendCCode
from unittest.mock import patch

import pytest


class TestParserANTLRPaths:
    """Test ANTLR-specific code paths."""

    def test_fallback_parser_used_when_antlr_missing(self):
        """Test that fallback parser is used when ANTLR artifacts are missing."""
        content = 'model_path = "test.tflite"\nquantize = int8'
        result = parse_edgeflow_string(content)
        assert result["model_path"] == "test.tflite"
        assert result["quantize"] == "int8"

    def test_parse_with_visitor_when_available(self):
        """Test visitor parsing when ANTLR artifacts are available."""
        # This test only runs if ANTLR artifacts exist
        try:
            from edgeflow.parser import EdgeFlowVisitor  # noqa: F401
        except ImportError:
            pytest.skip("ANTLR artifacts not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ef", delete=False) as f:
            f.write('model_path = "test.tflite"\nquantize = int8')
            f.flush()
            path = f.name

        try:
            # Force use of ANTLR path by mocking has_antlr
            import edgeflow.parser as parser

            original_has_antlr = parser.has_antlr
            parser.has_antlr = True

            result = parse_edgeflow_file(path)
            assert result["model_path"] == "test.tflite"
            assert result["quantize"] == "int8"

            parser.has_antlr = original_has_antlr
        finally:
            os.unlink(path)


class TestDay2APIExports:
    """Test Day 2 API export mechanisms."""

    def test_ensure_day2_exports_called(self):
        """Test that _ensure_day2_exports is called on import."""
        # Re-import to trigger export logic
        import importlib
        import edgeflow.parser as parser

        importlib.reload(parser)

        # Check that all required exports exist
        assert hasattr(parser, "EdgeFlowParserError")
        assert hasattr(parser, "parse_edgeflow_string")
        assert hasattr(parser, "parse_edgeflow_file")
        assert hasattr(parser, "validate_config")

    @pytest.mark.skip(reason="Internal method removed")
    def test_fallback_exports_when_parser_py_missing(self):
        """Test fallback exports when parser.py is not found."""
        with patch("os.path.isfile", return_value=False):
            import edgeflow.parser as parser

            # Force re-execution of _ensure_day2_exports
            parser._ensure_day2_exports()

            assert hasattr(parser, "EdgeFlowParserError")
            assert hasattr(parser, "parse_edgeflow_string")
            assert hasattr(parser, "parse_edgeflow_file")
            assert hasattr(parser, "validate_config")

    def test_fallback_validate_config(self):
        """Test fallback validate_config implementation."""
        # Test with valid config
        valid_config = {"model_path": "test.tflite", "quantize": "int8"}
        is_valid, errors = validate_config(valid_config)
        assert is_valid is True
        assert errors == []

        # Test with invalid config (missing model_path)
        invalid_config = {"quantize": "int8"}
        is_valid, errors = validate_config(invalid_config)
        assert is_valid is False
        assert len(errors) > 0
        assert "model_path" in errors[0]

        # Test with empty model_path
        empty_path_config = {"model_path": "", "quantize": "int8"}
        is_valid, errors = validate_config(empty_path_config)
        assert is_valid is False
        assert len(errors) > 0


class TestFallbackParser:
    """Test fallback parser specific functionality."""

    def test_fallback_parser_syntax_error_detection(self):
        """Test that fallback parser detects syntax errors."""
        from edgeflow.parser import EdgeFlowParserError, parse_edgeflow_string

        # Test multiple equals signs
        with pytest.raises(EdgeFlowParserError, match="syntax error"):
            parse_edgeflow_string("key === value")

        # Test line without proper value format (should pass in fallback)
        # Fallback parser accepts unquoted identifiers
        result = parse_edgeflow_string("key = identifier")
        assert result["key"] == "identifier"

    def test_fallback_parser_empty_lines_and_comments(self):
        """Test fallback parser handles empty lines and comments."""
        content = """
# This is a comment
model_path = "test.tflite"

# Another comment
quantize = int8
        """
        result = parse_edgeflow_string(content)
        assert result["model_path"] == "test.tflite"
        assert result["quantize"] == "int8"

    def test_fallback_parser_type_inference(self):
        """Test fallback parser type inference."""
        content = """
string_val = "test"
int_val = 42
float_val = 3.14
bool_true = true
bool_false = false
identifier = some_identifier
        """
        result = parse_edgeflow_string(content)
        assert result["string_val"] == "test"
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_true"] is True
        assert result["bool_false"] is False
        assert result["identifier"] == "some_identifier"


class TestEdgeFlowParserError:
    """Test custom exception class."""

    def test_parser_error_raised_on_invalid_syntax(self):
        """Test EdgeFlowParserError is raised for invalid syntax."""
        from edgeflow.parser import EdgeFlowParserError, parse_edgeflow_string

        with pytest.raises(EdgeFlowParserError) as exc_info:
            parse_edgeflow_string("invalid === syntax")
        assert "syntax error" in str(exc_info.value).lower()

    def test_parser_error_inheritance(self):
        """Test EdgeFlowParserError inherits from Exception."""
        assert issubclass(EdgeFlowParserError, Exception)


class TestParseEfLegacy:
    """Test legacy parse_ef function."""

    def test_parse_ef_function_exists(self):
        """Test that parse_ef function is available for backward compatibility."""
        from edgeflow.parser import parse_edgeflow_file as parse_ef

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ef", delete=False) as f:
            f.write('model_path = "legacy.tflite"')
            f.flush()
            path = f.name

        try:
            result = parse_ef(path)
            # Parser may return either 'model' or 'model_path' based on version
            assert (
                result.get("model") == "legacy.tflite"
                or result.get("model_path") == "legacy.tflite"
            )
        finally:
            os.unlink(path)

    def test_parse_ef_with_nonexistent_file(self):
        """Test parse_ef with non-existent file."""
        from edgeflow.parser import parse_edgeflow_file as parse_ef

        with pytest.raises(FileNotFoundError):
            parse_ef("/nonexistent/file.ef")


class TestModuleAttributes:
    """Test module-level attributes and exports."""

    def test_all_exports(self):
        """Test __all__ exports are correctly defined."""
        import edgeflow.parser as parser

        assert hasattr(parser, "__all__")
        expected_exports = [
            "parse_ef",
            "EdgeFlowParserError",
            "parse_edgeflow_file",
            "parse_edgeflow_string",
            "validate_config",
        ]
        for export in expected_exports:
            assert export in parser.__all__
            assert hasattr(parser, export)

    def test_type_checking_imports(self):
        """Test TYPE_CHECKING conditional imports."""
        # This mainly tests that the module loads without errors
        import edgeflow.parser as parser

        # These should be available after import
        assert callable(parser.parse_edgeflow_string)
        assert callable(parser.parse_edgeflow_file)
        assert callable(parser.validate_config)

    def test_c_identifier_sanitization(self):
        backend = EdgeBackendCCode()
        # Test hyphen and dot replacement
        assert backend._sanitize_identifier("layer-1.input") == "LAYER_1_INPUT"
        # Test leading digit (C identifiers cannot start with numbers)
        assert backend._sanitize_identifier("2nd_layer") == "_2ND_LAYER"
        # Test special characters
        assert backend._sanitize_identifier("node@link") == "NODE_LINK"
        # Test spaces
        assert backend._sanitize_identifier("final layer") == "FINAL_LAYER"