"""Tests for PII detection functionality."""

import pytest

from aigentic.core.pii_detector import PIIDetector, has_pii


class TestPIIDetector:
    """Test suite for PII detection."""

    def test_ssn_detection(self):
        """Test SSN pattern detection."""
        # Valid SSN formats
        assert PIIDetector.has_pii("My SSN is 123-45-6789")
        assert PIIDetector.has_pii("SSN: 987-65-4321")

        # Invalid formats should not match
        assert not PIIDetector.has_pii("My SSN is 12-345-6789")  # Wrong format
        assert not PIIDetector.has_pii("Number: 12345678")  # Too short, no dashes

    def test_credit_card_detection(self):
        """Test credit card pattern detection."""
        # Valid credit card formats
        assert PIIDetector.has_pii("Card: 1234-5678-9012-3456")
        assert PIIDetector.has_pii("Card: 1234 5678 9012 3456")
        assert PIIDetector.has_pii("Card: 1234567890123456")

        # Invalid formats
        assert not PIIDetector.has_pii("Card: 123-456-789")  # Too short

    def test_email_detection(self):
        """Test email pattern detection."""
        # Valid emails
        assert PIIDetector.has_pii("Contact me at john.doe@example.com")
        assert PIIDetector.has_pii("Email: user+tag@domain.co.uk")
        assert PIIDetector.has_pii("test_user@sub.example.org")

        # Invalid emails
        assert not PIIDetector.has_pii("Not an email: @example.com")
        assert not PIIDetector.has_pii("Also not: user@")

    def test_phone_detection(self):
        """Test phone number pattern detection."""
        # Valid phone formats
        assert PIIDetector.has_pii("Call me at 555-123-4567")
        assert PIIDetector.has_pii("Phone: 555.123.4567")
        assert PIIDetector.has_pii("Mobile: 5551234567")

        # Invalid formats
        assert not PIIDetector.has_pii("Phone: 123-45")  # Too short

    def test_multiple_pii_types(self):
        """Test detection of multiple PII types in one text."""
        text = "My SSN is 123-45-6789 and email is user@example.com"
        detected = PIIDetector.detect(text)

        assert "ssn" in detected
        assert "email" in detected
        assert len(detected["ssn"]) == 1
        assert len(detected["email"]) == 1

    def test_no_pii(self):
        """Test text without PII."""
        clean_text = "This is a normal message without any sensitive information."
        assert not PIIDetector.has_pii(clean_text)
        assert PIIDetector.detect(clean_text) == {}

    def test_has_pii_convenience_function(self):
        """Test the convenience has_pii function."""
        assert has_pii("SSN: 123-45-6789")
        assert not has_pii("No sensitive data here")

    def test_pii_detection_boundary(self):
        """Test that PII detection respects word boundaries."""
        # Should detect standalone SSN
        assert PIIDetector.has_pii("123-45-6789")

        # Should not detect if it's part of a longer string (depending on boundary)
        # The regex uses \b so it should still detect at word boundaries

    def test_detect_returns_matches(self):
        """Test that detect() returns the actual matched values."""
        text = "Contact: user@example.com or admin@test.org"
        detected = PIIDetector.detect(text)

        assert "email" in detected
        assert "user@example.com" in detected["email"]
        assert "admin@test.org" in detected["email"]

    def test_credit_card_various_separators(self):
        """Test credit card detection with different separators."""
        assert PIIDetector.has_pii("1234-5678-9012-3456")
        assert PIIDetector.has_pii("1234 5678 9012 3456")
        assert PIIDetector.has_pii("1234567890123456")

    def test_phone_various_separators(self):
        """Test phone detection with different separators."""
        assert PIIDetector.has_pii("555-123-4567")
        assert PIIDetector.has_pii("555.123.4567")
        assert PIIDetector.has_pii("5551234567")
