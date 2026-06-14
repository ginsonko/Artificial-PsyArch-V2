from __future__ import annotations

"""
External education systems for APV2.1.

This package deliberately sits outside ``core``. A teacher may suggest,
reward, punish, or demonstrate, but AP core remains responsible for cognition,
action selection, actuator execution, and learning from feedback.
"""

from education.intervention import EducationInterventionBuffer, normalize_education_intervention
from education.fake_llm_teacher import FakeLLMBuildingBlockTeacher, FakeLLMNoisyGeneralizationTeacher
from education.llm_teacher import LLMTeacherClient, LLMTeacherConfig
from education.skill_scaffold import SkillScaffoldController
from education.skill_protocol_v2 import SkillScaffoldProtocolV2Controller, normalize_skill_scaffold_spec_v2

__all__ = [
    "EducationInterventionBuffer",
    "FakeLLMBuildingBlockTeacher",
    "FakeLLMNoisyGeneralizationTeacher",
    "LLMTeacherClient",
    "LLMTeacherConfig",
    "SkillScaffoldController",
    "SkillScaffoldProtocolV2Controller",
    "normalize_education_intervention",
    "normalize_skill_scaffold_spec_v2",
]
