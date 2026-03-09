class SentimentAPIError(Exception):
    """Base exception for sentiment API."""

    pass


class ModelLoadError(SentimentAPIError):
    """Raised when model fails to load."""

    pass


class PredictionError(SentimentAPIError):
    """Raised when prediction fails."""

    pass


class TokenizationError(SentimentAPIError):
    """Raised when tokenization fails."""

    pass
