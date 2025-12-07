"""
FPF Validation Layer

Validates reasoning outputs against FPF principles.

Key validations:
- A.7 Strict Distinction: Prevent category conflation
- B.3 Trust: F-G-R must be computed, not intuited
- B.5 ADI: Phase transitions must follow gates
- A.1.1 Context: Terms must be used within declared context
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


# =============================================================================
# STRICT DISTINCTIONS (A.7)
# =============================================================================

class DistinctionCategory(Enum):
    """The 8 core distinctions in FPF that must NEVER be conflated."""
    SYSTEM = "System"           # Physical/operational holon
    EPISTEME = "Episteme"       # Knowledge artifact
    ROLE = "Role"               # Contextual responsibility (abstract)
    HOLDER = "Holder"           # Entity that enacts a role
    METHOD = "Method"           # Abstract way of doing
    METHOD_DESC = "MethodDescription"  # Recipe/specification
    WORK = "Work"               # Actual execution record
    DESIGN_TIME = "DesignTime"  # Planning, specification
    RUN_TIME = "RunTime"        # Actual execution
    OBJECT = "Object"           # The thing itself
    DESCRIPTION = "Description" # Representation of the thing
    INTENSION = "Intension"     # Defining characteristics
    EXTENSION = "Extension"     # Set of instances
    CONTEXT = "Context"         # Frame of reference
    CONTENT = "Content"         # What's inside the frame


# The core distinction pairs - these categories must NEVER be conflated
STRICT_DISTINCTION_PAIRS = [
    (DistinctionCategory.SYSTEM, DistinctionCategory.EPISTEME,
     "System ≠ Episteme: Physical/operational holons vs knowledge artifacts"),
    (DistinctionCategory.ROLE, DistinctionCategory.HOLDER,
     "Role ≠ Holder: Contextual responsibility vs entity that enacts it"),
    (DistinctionCategory.METHOD, DistinctionCategory.METHOD_DESC,
     "Method ≠ MethodDescription: Abstract way vs recipe/specification"),
    (DistinctionCategory.METHOD_DESC, DistinctionCategory.WORK,
     "MethodDescription ≠ Work: Recipe vs actual execution record"),
    (DistinctionCategory.DESIGN_TIME, DistinctionCategory.RUN_TIME,
     "Design-time ≠ Run-time: Planning vs execution (chimera risk)"),
    (DistinctionCategory.OBJECT, DistinctionCategory.DESCRIPTION,
     "Object ≠ Description: The thing vs its representation"),
    (DistinctionCategory.INTENSION, DistinctionCategory.EXTENSION,
     "Intension ≠ Extension: Defining characteristics vs instances"),
    (DistinctionCategory.CONTEXT, DistinctionCategory.CONTENT,
     "Context ≠ Content: Frame of reference vs what's inside"),
]


# Keyword patterns that may indicate category conflation
CONFLATION_PATTERNS = {
    ("system", "episteme"): [
        "system knowledge", "system knows", "system understands",
        "knowledge system",  # OK if referring to KD-CAL
    ],
    ("role", "holder"): [
        "role is", "the role performs", "role does",
        "person role",  # OK
    ],
    ("method", "work"): [
        "method executed", "method ran", "method completed",
        "work method",  # OK
    ],
    ("design", "run"): [
        "design runs", "design executes", "runtime design",
        "design-time behavior",  # Anti-pattern warning
    ],
    ("object", "description"): [
        "object says", "object describes", "description is the",
        "the model is the system",  # Classic anti-pattern
    ],
}


# =============================================================================
# VIOLATION TYPES
# =============================================================================

class ViolationSeverity(Enum):
    """Severity levels for FPF violations."""
    CRITICAL = auto()    # Must fix before proceeding
    IMPORTANT = auto()   # Should fix, may cause issues
    WARNING = auto()     # Style/clarity issue
    INFO = auto()        # Note for improvement


@dataclass
class FPFViolation:
    """A detected violation of FPF principles."""
    violation_id: str
    category: str  # Which FPF principle violated
    severity: ViolationSeverity
    description: str
    location: str  # Where in the reasoning this occurred
    suggestion: str  # How to fix
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "violation_id": self.violation_id,
            "category": self.category,
            "severity": self.severity.name,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validating a reasoning step or output."""
    is_valid: bool
    violations: list[FPFViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fpf_coverage: dict[str, bool] = field(default_factory=dict)

    @property
    def critical_violations(self) -> list[FPFViolation]:
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]

    @property
    def has_critical(self) -> bool:
        return len(self.critical_violations) > 0

    def summary(self) -> str:
        if self.is_valid:
            return f"Valid (warnings: {len(self.warnings)})"
        return f"Invalid ({len(self.critical_violations)} critical, {len(self.violations)} total violations)"


# =============================================================================
# STRICT DISTINCTION VALIDATOR
# =============================================================================

class StrictDistinctionValidator:
    """
    A.7: Validates that reasoning does not conflate distinct categories.

    The 8 core distinctions are foundational to FPF. Conflating them
    leads to category errors that propagate through all reasoning.
    """

    def __init__(self):
        self._violation_counter = 0

    def validate_text(self, text: str, location: str = "unknown") -> list[FPFViolation]:
        """
        Check text for potential category conflations.

        This is a heuristic check - it may have false positives.
        """
        violations = []
        text_lower = text.lower()

        for (cat1, cat2), patterns in CONFLATION_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    violations.append(self._create_violation(
                        category="A.7_StrictDistinction",
                        severity=ViolationSeverity.WARNING,
                        description=f"Potential {cat1}/{cat2} conflation: '{pattern}'",
                        location=location,
                        suggestion=f"Verify that {cat1} and {cat2} are properly distinguished",
                    ))

        # Check for classic anti-patterns
        anti_patterns = [
            ("the model is the system", "Object/Description conflation - model describes, is not identical to, system"),
            ("the role is", "Role/Holder conflation - roles are enacted by holders, not identical to them"),
            ("method completed", "Method/Work conflation - methods are abstract, work is actual execution"),
        ]

        for pattern, desc in anti_patterns:
            if pattern in text_lower:
                violations.append(self._create_violation(
                    category="A.7_StrictDistinction",
                    severity=ViolationSeverity.IMPORTANT,
                    description=desc,
                    location=location,
                    suggestion="Use precise language that maintains FPF distinctions",
                ))

        return violations

    def validate_object_of_talk(
        self,
        category: str,
        description: str,
        location: str = "object_of_talk"
    ) -> list[FPFViolation]:
        """
        Validate that ObjectOfTalk category matches its description.

        Checks for mismatches between declared category and content.
        """
        violations = []
        description_lower = description.lower()

        # Category-specific checks
        if category == "System":
            episteme_indicators = ["theory", "model", "knowledge", "understanding", "belief"]
            for indicator in episteme_indicators:
                if indicator in description_lower and "system" not in description_lower:
                    violations.append(self._create_violation(
                        category="A.7_StrictDistinction",
                        severity=ViolationSeverity.IMPORTANT,
                        description=f"Category 'System' but description suggests Episteme ('{indicator}')",
                        location=location,
                        suggestion="If this is about knowledge/theory, use category 'Episteme'",
                    ))

        elif category == "Episteme":
            system_indicators = ["physical", "operational", "hardware", "device", "machine"]
            for indicator in system_indicators:
                if indicator in description_lower:
                    violations.append(self._create_violation(
                        category="A.7_StrictDistinction",
                        severity=ViolationSeverity.IMPORTANT,
                        description=f"Category 'Episteme' but description suggests System ('{indicator}')",
                        location=location,
                        suggestion="If this is about physical/operational entity, use category 'System'",
                    ))

        elif category == "Role":
            if any(name in description_lower for name in ["john", "alice", "bob", "the person", "the user"]):
                violations.append(self._create_violation(
                    category="A.7_StrictDistinction",
                    severity=ViolationSeverity.IMPORTANT,
                    description="Category 'Role' but description names a specific person (Holder)",
                    location=location,
                    suggestion="Roles are abstract responsibilities, not specific people. Use separate Holder reference.",
                ))

        return violations

    def validate_temporal_stance(
        self,
        declared_scope: str,
        reasoning_text: str,
        location: str = "temporal_stance"
    ) -> list[FPFViolation]:
        """
        Validate that temporal scope is consistently applied.

        Detects "chimeras" - mixing design-time and run-time in same entity.
        """
        violations = []
        text_lower = reasoning_text.lower()

        if declared_scope == "design_time":
            runtime_indicators = ["currently running", "at runtime", "executing now", "live"]
            for indicator in runtime_indicators:
                if indicator in text_lower:
                    violations.append(self._create_violation(
                        category="A.4_TemporalDuality",
                        severity=ViolationSeverity.CRITICAL,
                        description=f"Declared design-time but text suggests runtime ('{indicator}')",
                        location=location,
                        suggestion="Chimera detected - clearly separate design-time planning from runtime execution",
                    ))

        elif declared_scope == "run_time":
            design_indicators = ["will be designed", "planning to", "specification says", "in the design"]
            for indicator in design_indicators:
                if indicator in text_lower:
                    violations.append(self._create_violation(
                        category="A.4_TemporalDuality",
                        severity=ViolationSeverity.WARNING,
                        description=f"Declared runtime but text references design-time ('{indicator}')",
                        location=location,
                        suggestion="Keep runtime analysis focused on actual execution, not design specs",
                    ))

        return violations

    def _create_violation(
        self,
        category: str,
        severity: ViolationSeverity,
        description: str,
        location: str,
        suggestion: str,
    ) -> FPFViolation:
        self._violation_counter += 1
        return FPFViolation(
            violation_id=f"v_{self._violation_counter:04d}",
            category=category,
            severity=severity,
            description=description,
            location=location,
            suggestion=suggestion,
        )


# =============================================================================
# FGR VALIDATOR
# =============================================================================

class FGRValidator:
    """
    B.3: Validates that F-G-R assessments are properly computed.

    Key principle: "Trust is computed, not intuited."
    """

    def __init__(self):
        self._violation_counter = 0

    def validate_fgr_assessment(
        self,
        formality: int,
        reliability: float,
        evidence_count: int,
        location: str = "fgr_assessment"
    ) -> list[FPFViolation]:
        """Validate F-G-R assessment is properly computed."""
        violations = []

        # Check: reliability without evidence
        if reliability > 0.5 and evidence_count == 0:
            violations.append(FPFViolation(
                violation_id=f"fgr_{self._violation_counter}",
                category="B.3_TrustCalculus",
                severity=ViolationSeverity.CRITICAL,
                description=f"High reliability ({reliability}) claimed without evidence",
                location=location,
                suggestion="Reliability must be computed from evidence. Add evidence or lower reliability.",
            ))
            self._violation_counter += 1

        # Check: high formality claim without justification
        if formality >= 7 and evidence_count < 2:
            violations.append(FPFViolation(
                violation_id=f"fgr_{self._violation_counter}",
                category="B.3_TrustCalculus",
                severity=ViolationSeverity.IMPORTANT,
                description=f"High formality (F{formality}) claimed with insufficient evidence",
                location=location,
                suggestion=f"F{formality} (machine-verified) requires strong evidence chain",
            ))
            self._violation_counter += 1

        # Check: reliability = 1.0 (should be very rare)
        if reliability == 1.0:
            violations.append(FPFViolation(
                violation_id=f"fgr_{self._violation_counter}",
                category="B.3_TrustCalculus",
                severity=ViolationSeverity.WARNING,
                description="Perfect reliability (1.0) claimed - this is extremely rare",
                location=location,
                suggestion="Only mathematical axioms and fully verified proofs should have R=1.0",
            ))
            self._violation_counter += 1

        return violations


# =============================================================================
# COMPREHENSIVE VALIDATOR
# =============================================================================

class FPFReasoningValidator:
    """
    Comprehensive validator for FPF reasoning outputs.

    Combines all validation checks.
    """

    def __init__(self):
        self.distinction_validator = StrictDistinctionValidator()
        self.fgr_validator = FGRValidator()

    def validate_reasoning_step(self, step: dict) -> ValidationResult:
        """
        Validate a complete reasoning step.

        Args:
            step: Dictionary representation of FPFReasoningStep
        """
        violations = []
        warnings = []
        fpf_coverage = {}

        # 1. Validate Strict Distinction
        if "object_of_talk" in step:
            obj = step["object_of_talk"]
            violations.extend(self.distinction_validator.validate_object_of_talk(
                category=obj.get("category", ""),
                description=obj.get("description", ""),
                location="object_of_talk",
            ))
            fpf_coverage["A.7_StrictDistinction"] = True

        # 2. Validate Temporal Stance
        if "temporal_stance" in step and "current_understanding" in step:
            ts = step["temporal_stance"]
            violations.extend(self.distinction_validator.validate_temporal_stance(
                declared_scope=ts.get("scope", ""),
                reasoning_text=step.get("current_understanding", ""),
                location="temporal_stance",
            ))
            fpf_coverage["A.4_TemporalDuality"] = True

        # 3. Validate F-G-R Assessment
        if "trust_assessment" in step and step["trust_assessment"]:
            trust = step["trust_assessment"]
            violations.extend(self.fgr_validator.validate_fgr_assessment(
                formality=trust.get("formality", 0),
                reliability=trust.get("reliability", 0),
                evidence_count=len(trust.get("evidence_references", [])),
                location="trust_assessment",
            ))
            fpf_coverage["B.3_TrustCalculus"] = True

        # 4. Check for general text issues
        understanding = step.get("current_understanding", "")
        violations.extend(self.distinction_validator.validate_text(
            understanding,
            location="current_understanding",
        ))

        # Determine validity
        has_critical = any(v.severity == ViolationSeverity.CRITICAL for v in violations)

        return ValidationResult(
            is_valid=not has_critical,
            violations=violations,
            warnings=warnings,
            fpf_coverage=fpf_coverage,
        )

    def validate_response(self, response: dict) -> ValidationResult:
        """Validate a final FPFResponse."""
        violations = []
        warnings = []

        response_text = response.get("response", "")

        # Check response text for category conflations
        violations.extend(self.distinction_validator.validate_text(
            response_text,
            location="response",
        ))

        # Check confidence matches reasoning
        confidence = response.get("confidence", "")
        trace = response.get("reasoning_trace", [])

        if confidence == "high" and len(trace) < 2:
            warnings.append("High confidence claimed with minimal reasoning trace")

        return ValidationResult(
            is_valid=len([v for v in violations if v.severity == ViolationSeverity.CRITICAL]) == 0,
            violations=violations,
            warnings=warnings,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_distinction_guidance() -> str:
    """Get guidance text on FPF strict distinctions."""
    lines = ["## FPF Strict Distinctions (A.7)\n"]
    lines.append("These category pairs must NEVER be conflated:\n")

    for cat1, cat2, description in STRICT_DISTINCTION_PAIRS:
        lines.append(f"- **{cat1.value} ≠ {cat2.value}**: {description.split(': ')[1]}")

    lines.append("\n### Why This Matters")
    lines.append("Category conflation is the root cause of many reasoning errors.")
    lines.append("When you mix System/Episteme, you get confused thinking.")
    lines.append("When you mix Role/Holder, you lose track of responsibilities.")
    lines.append("When you mix Design-time/Run-time, you create 'chimeras' that don't work.")

    return "\n".join(lines)
