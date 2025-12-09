"""Custom exceptions for ACDP."""


class ACDPError(Exception):
    """Base exception for all ACDP errors."""

    pass


class DiscoveryError(ACDPError):
    """Error during capability discovery."""

    pass


class RegistrationError(ACDPError):
    """Error during agent registration."""

    pass


class ExecutionError(ACDPError):
    """Error during agent execution."""

    pass


class NegotiationError(ACDPError):
    """Error during capability negotiation."""

    pass


class ValidationError(ACDPError):
    """Error during capability validation."""

    pass


class TransportError(ACDPError):
    """Error in transport layer."""

    pass


class AuthenticationError(ACDPError):
    """Error during authentication."""

    pass


class TrustError(ACDPError):
    """Error in trust/reputation system."""

    pass


class CapabilityNotFoundError(DiscoveryError):
    """Requested capability not found."""

    pass


class IncompatibleCapabilityError(NegotiationError):
    """Capability versions or specifications are incompatible."""

    pass


class NegotiationTimeoutError(NegotiationError):
    """Negotiation timed out."""

    pass


class NegotiationRejectedError(NegotiationError):
    """Negotiation was rejected by the other party."""

    pass


class ValidationFailedError(ValidationError):
    """Capability validation failed."""

    pass


class UntrustedAgentError(TrustError):
    """Agent does not meet trust requirements."""

    pass


class TransportNotAvailableError(TransportError):
    """Requested transport protocol is not available."""

    pass


class MessageSerializationError(TransportError):
    """Error serializing or deserializing message."""

    pass


class InvalidDIDError(ACDPError):
    """Invalid decentralized identifier (DID)."""

    pass


class InvalidCapabilityError(ACDPError):
    """Invalid capability specification."""

    pass
