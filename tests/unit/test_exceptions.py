"""Tests for custom exceptions."""

import pytest

from capabilitymesh.core.exceptions import (
    ACDPError,
    AuthenticationError,
    CapabilityNotFoundError,
    DiscoveryError,
    ExecutionError,
    IncompatibleCapabilityError,
    InvalidCapabilityError,
    InvalidDIDError,
    MessageSerializationError,
    NegotiationError,
    NegotiationRejectedError,
    NegotiationTimeoutError,
    RegistrationError,
    TransportError,
    TransportNotAvailableError,
    TrustError,
    UntrustedAgentError,
    ValidationError,
    ValidationFailedError,
)


class TestBaseException:
    """Test ACDPError base exception."""

    def test_acdp_error_can_be_raised(self):
        """Test that ACDPError can be raised."""
        with pytest.raises(ACDPError):
            raise ACDPError("Test error")

    def test_acdp_error_with_message(self):
        """Test ACDPError with custom message."""
        error = ACDPError("Custom error message")
        assert str(error) == "Custom error message"

    def test_acdp_error_without_message(self):
        """Test ACDPError without message."""
        error = ACDPError()
        assert str(error) == ""

    def test_acdp_error_is_exception(self):
        """Test that ACDPError inherits from Exception."""
        assert issubclass(ACDPError, Exception)


class TestPrimaryExceptions:
    """Test primary exception classes."""

    def test_discovery_error_inheritance(self):
        """Test DiscoveryError inherits from ACDPError."""
        assert issubclass(DiscoveryError, ACDPError)
        assert issubclass(DiscoveryError, Exception)

    def test_discovery_error_can_be_raised(self):
        """Test that DiscoveryError can be raised."""
        with pytest.raises(DiscoveryError):
            raise DiscoveryError("Discovery failed")

    def test_registration_error_inheritance(self):
        """Test RegistrationError inherits from ACDPError."""
        assert issubclass(RegistrationError, ACDPError)
        assert issubclass(RegistrationError, Exception)

    def test_registration_error_can_be_raised(self):
        """Test that RegistrationError can be raised."""
        with pytest.raises(RegistrationError):
            raise RegistrationError("Registration failed")

    def test_execution_error_inheritance(self):
        """Test ExecutionError inherits from ACDPError."""
        assert issubclass(ExecutionError, ACDPError)
        assert issubclass(ExecutionError, Exception)

    def test_execution_error_can_be_raised(self):
        """Test that ExecutionError can be raised."""
        with pytest.raises(ExecutionError):
            raise ExecutionError("Execution failed")

    def test_negotiation_error_inheritance(self):
        """Test NegotiationError inherits from ACDPError."""
        assert issubclass(NegotiationError, ACDPError)
        assert issubclass(NegotiationError, Exception)

    def test_negotiation_error_can_be_raised(self):
        """Test that NegotiationError can be raised."""
        with pytest.raises(NegotiationError):
            raise NegotiationError("Negotiation failed")

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from ACDPError."""
        assert issubclass(ValidationError, ACDPError)
        assert issubclass(ValidationError, Exception)

    def test_validation_error_can_be_raised(self):
        """Test that ValidationError can be raised."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")

    def test_transport_error_inheritance(self):
        """Test TransportError inherits from ACDPError."""
        assert issubclass(TransportError, ACDPError)
        assert issubclass(TransportError, Exception)

    def test_transport_error_can_be_raised(self):
        """Test that TransportError can be raised."""
        with pytest.raises(TransportError):
            raise TransportError("Transport failed")

    def test_authentication_error_inheritance(self):
        """Test AuthenticationError inherits from ACDPError."""
        assert issubclass(AuthenticationError, ACDPError)
        assert issubclass(AuthenticationError, Exception)

    def test_authentication_error_can_be_raised(self):
        """Test that AuthenticationError can be raised."""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Authentication failed")

    def test_trust_error_inheritance(self):
        """Test TrustError inherits from ACDPError."""
        assert issubclass(TrustError, ACDPError)
        assert issubclass(TrustError, Exception)

    def test_trust_error_can_be_raised(self):
        """Test that TrustError can be raised."""
        with pytest.raises(TrustError):
            raise TrustError("Trust error")


class TestSpecializedExceptions:
    """Test specialized exception subclasses."""

    def test_capability_not_found_error_inheritance(self):
        """Test CapabilityNotFoundError inherits from DiscoveryError."""
        assert issubclass(CapabilityNotFoundError, DiscoveryError)
        assert issubclass(CapabilityNotFoundError, ACDPError)

    def test_capability_not_found_error_can_be_raised(self):
        """Test that CapabilityNotFoundError can be raised."""
        with pytest.raises(CapabilityNotFoundError):
            raise CapabilityNotFoundError("Capability 'translate' not found")

    def test_capability_not_found_caught_as_discovery_error(self):
        """Test that CapabilityNotFoundError can be caught as DiscoveryError."""
        with pytest.raises(DiscoveryError):
            raise CapabilityNotFoundError("Capability not found")

    def test_incompatible_capability_error_inheritance(self):
        """Test IncompatibleCapabilityError inherits from NegotiationError."""
        assert issubclass(IncompatibleCapabilityError, NegotiationError)
        assert issubclass(IncompatibleCapabilityError, ACDPError)

    def test_incompatible_capability_error_can_be_raised(self):
        """Test that IncompatibleCapabilityError can be raised."""
        with pytest.raises(IncompatibleCapabilityError):
            raise IncompatibleCapabilityError("Incompatible versions")

    def test_negotiation_timeout_error_inheritance(self):
        """Test NegotiationTimeoutError inherits from NegotiationError."""
        assert issubclass(NegotiationTimeoutError, NegotiationError)
        assert issubclass(NegotiationTimeoutError, ACDPError)

    def test_negotiation_timeout_error_can_be_raised(self):
        """Test that NegotiationTimeoutError can be raised."""
        with pytest.raises(NegotiationTimeoutError):
            raise NegotiationTimeoutError("Negotiation timed out after 30s")

    def test_negotiation_rejected_error_inheritance(self):
        """Test NegotiationRejectedError inherits from NegotiationError."""
        assert issubclass(NegotiationRejectedError, NegotiationError)
        assert issubclass(NegotiationRejectedError, ACDPError)

    def test_negotiation_rejected_error_can_be_raised(self):
        """Test that NegotiationRejectedError can be raised."""
        with pytest.raises(NegotiationRejectedError):
            raise NegotiationRejectedError("Agent rejected negotiation")

    def test_validation_failed_error_inheritance(self):
        """Test ValidationFailedError inherits from ValidationError."""
        assert issubclass(ValidationFailedError, ValidationError)
        assert issubclass(ValidationFailedError, ACDPError)

    def test_validation_failed_error_can_be_raised(self):
        """Test that ValidationFailedError can be raised."""
        with pytest.raises(ValidationFailedError):
            raise ValidationFailedError("Schema validation failed")

    def test_untrusted_agent_error_inheritance(self):
        """Test UntrustedAgentError inherits from TrustError."""
        assert issubclass(UntrustedAgentError, TrustError)
        assert issubclass(UntrustedAgentError, ACDPError)

    def test_untrusted_agent_error_can_be_raised(self):
        """Test that UntrustedAgentError can be raised."""
        with pytest.raises(UntrustedAgentError):
            raise UntrustedAgentError("Agent trust level too low")

    def test_transport_not_available_error_inheritance(self):
        """Test TransportNotAvailableError inherits from TransportError."""
        assert issubclass(TransportNotAvailableError, TransportError)
        assert issubclass(TransportNotAvailableError, ACDPError)

    def test_transport_not_available_error_can_be_raised(self):
        """Test that TransportNotAvailableError can be raised."""
        with pytest.raises(TransportNotAvailableError):
            raise TransportNotAvailableError("HTTP transport not available")

    def test_message_serialization_error_inheritance(self):
        """Test MessageSerializationError inherits from TransportError."""
        assert issubclass(MessageSerializationError, TransportError)
        assert issubclass(MessageSerializationError, ACDPError)

    def test_message_serialization_error_can_be_raised(self):
        """Test that MessageSerializationError can be raised."""
        with pytest.raises(MessageSerializationError):
            raise MessageSerializationError("Failed to serialize JSON")


class TestMiscExceptions:
    """Test miscellaneous exception classes."""

    def test_invalid_did_error_inheritance(self):
        """Test InvalidDIDError inherits from ACDPError."""
        assert issubclass(InvalidDIDError, ACDPError)
        assert issubclass(InvalidDIDError, Exception)

    def test_invalid_did_error_can_be_raised(self):
        """Test that InvalidDIDError can be raised."""
        with pytest.raises(InvalidDIDError):
            raise InvalidDIDError("DID format invalid")

    def test_invalid_capability_error_inheritance(self):
        """Test InvalidCapabilityError inherits from ACDPError."""
        assert issubclass(InvalidCapabilityError, ACDPError)
        assert issubclass(InvalidCapabilityError, Exception)

    def test_invalid_capability_error_can_be_raised(self):
        """Test that InvalidCapabilityError can be raised."""
        with pytest.raises(InvalidCapabilityError):
            raise InvalidCapabilityError("Invalid capability specification")


class TestExceptionCatching:
    """Test exception catching behavior."""

    def test_catch_specialized_as_base(self):
        """Test that specialized exceptions can be caught as base exception."""
        with pytest.raises(ACDPError):
            raise CapabilityNotFoundError("Not found")

    def test_catch_specialized_as_parent(self):
        """Test that specialized exceptions can be caught as parent exception."""
        with pytest.raises(NegotiationError):
            raise NegotiationTimeoutError("Timeout")

    def test_multiple_inheritance_levels(self):
        """Test catching at different inheritance levels."""
        error = UntrustedAgentError("Low trust")

        # Should be caught at all levels
        assert isinstance(error, UntrustedAgentError)
        assert isinstance(error, TrustError)
        assert isinstance(error, ACDPError)
        assert isinstance(error, Exception)

    def test_exception_messages_preserved(self):
        """Test that exception messages are preserved through inheritance."""
        message = "Detailed error information"

        errors = [
            DiscoveryError(message),
            CapabilityNotFoundError(message),
            NegotiationTimeoutError(message),
            ValidationFailedError(message),
        ]

        for error in errors:
            assert str(error) == message


class TestExceptionUsagePatterns:
    """Test common exception usage patterns."""

    def test_exception_with_formatted_message(self):
        """Test exceptions with formatted messages."""
        agent_id = "agent-123"
        capability = "translate"
        error = CapabilityNotFoundError(
            f"Agent {agent_id} does not have capability '{capability}'"
        )
        assert agent_id in str(error)
        assert capability in str(error)

    def test_exception_chaining(self):
        """Test exception chaining with 'from' clause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ExecutionError("Execution failed") from e
        except ExecutionError as ex:
            assert ex.__cause__ is not None
            assert isinstance(ex.__cause__, ValueError)
            assert str(ex.__cause__) == "Original error"

    def test_exception_with_no_message_is_empty_string(self):
        """Test that exceptions without messages have empty string representation."""
        errors = [
            ACDPError(),
            DiscoveryError(),
            ExecutionError(),
            NegotiationError(),
        ]

        for error in errors:
            assert str(error) == ""

    def test_reraise_as_different_type(self):
        """Test re-raising an exception as a different type."""
        original_message = "Connection refused"

        with pytest.raises(TransportError) as exc_info:
            try:
                raise ConnectionError(original_message)
            except ConnectionError as e:
                raise TransportError(f"Transport failed: {e}") from e

        assert "Transport failed" in str(exc_info.value)
        assert original_message in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ConnectionError)
