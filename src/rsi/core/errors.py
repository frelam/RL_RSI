class RsiError(Exception):
    """Base error for the framework."""


class ConfigError(RsiError):
    """Invalid or missing configuration."""


class RegistryError(RsiError):
    """Unknown plugin or failed plugin construction."""


class AdapterError(RsiError):
    """An external adapter failed or returned an invalid payload."""


class SchemaError(RsiError):
    """Structured output did not match the declared schema."""
