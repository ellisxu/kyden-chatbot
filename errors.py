from typing import Optional, Any

POLICY_VIOLATION_ERROR_CODE: int = 10001


class BaseError(Exception):
    code: int = 0
    message: Optional[str] = None

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code
        self.message = message


class PolicyViolationError(BaseError):
    """Raised when the content policy is violated."""

    def __init__(self, message):
        super().__init__(message=message, code=POLICY_VIOLATION_ERROR_CODE)
