# AP Learned Skill Registry Report

日期: 2026-06-03

## 结论

已经通过 AP-Core 证据的底层数学技能可以固化为 `action::skill.*` 行动入口。高层任务可以复用这些入口, 但报告必须声明依赖, 且不能把复用实验说成从零证明。

## 技能依赖图

```text
skill.math.count_successor.v1
  -> skill.math.ten_within_add_sub.v1
      -> skill.math.vertical_add.v1
      -> skill.math.vertical_subtract.v1
  -> skill.math.small_mul_div.v1
      -> skill.math.vertical_multiply.v1
  -> skill.math.vertical_division_one_digit.v1
  -> skill.math.repeated_subtract_division.v1
      -> skill.math.linear_equation_word_problem.v1
```

## 已注册技能

### skill.math.count_successor.v1

- action_id: `action::skill.math.count_successor.v1`
- title: Count successor / predecessor and small object quantity relation
- source_package: `experiments/countloop0_object_quantity_successor_acquisition/countloop0_object_quantity_experience_package.json`
- package_sha256: `683162379967A9134DD2D4C0D65820E0CB7E5201580437435C8DBF1262229CAD`
- pytest: `python -m pytest tests/test_countloop0_object_quantity_successor_acquisition.py -q`
- dependencies: `none`
- allowed_inputs: small controlled quantity/add-remove states within the demonstrated CountLoop-0 boundary
- abstain_policy: abstain/hold when the object quantity relation is outside learned small-range evidence
- evidence_report: `experiments/countloop0_object_quantity_successor_acquisition/APV21_CountLoop0_object_quantity_successor_acquisition.md`

### skill.math.ten_within_add_sub.v1

- action_id: `action::skill.math.ten_within_add_sub.v1`
- title: Ten-within add/sub acquired fact and process skill
- source_package: `experiments/elementary_math_nosolver0_add_sub_acquired/elementary_math_nosolver0_skill_package.json`
- package_sha256: `20AB0811D11CEB450B4D58A7D381FD6E78441F6BD3A0A1F7FECFDF0E7741BE2C`
- pytest: `python -m pytest tests/test_elementary_math_nosolver0_add_sub_acquired.py -q`
- dependencies: `skill.math.count_successor.v1`
- allowed_inputs: controlled ten-within add/sub slices covered by NoSolver0
- abstain_policy: abstain when fact/process confidence is below threshold or outside covered range
- evidence_report: `experiments/elementary_math_nosolver0_add_sub_acquired/APV21_ElementaryMath_NoSolver0_add_sub_acquired.md`

### skill.math.small_mul_div.v1

- action_id: `action::skill.math.small_mul_div.v1`
- title: Small-range equal-group multiplication/division skill
- source_package: `experiments/elementary_math_nosolver1_mul_div_acquired/elementary_math_nosolver1_skill_package.json`
- package_sha256: `9FFD9A69721CBFE5DBE257053152A04C25884072498DAB60B957C6A13AB5CBD0`
- pytest: `python -m pytest tests/test_elementary_math_nosolver1_mul_div_acquired.py -q`
- dependencies: `skill.math.count_successor.v1, skill.math.ten_within_add_sub.v1`
- allowed_inputs: small controlled equal-group multiplication/division slices covered by NoSolver1
- abstain_policy: abstain when divisor/group relation is outside learned process boundary
- evidence_report: `experiments/elementary_math_nosolver1_mul_div_acquired/APV21_ElementaryMath_NoSolver1_mul_div_acquired.md`

### skill.math.vertical_add.v1

- action_id: `action::skill.math.vertical_add.v1`
- title: Vertical addition process skill
- source_package: `experiments/elementary_math_nosolver2_vertical_add_sub_acquired/elementary_math_nosolver2_skill_package.json`
- package_sha256: `E89F04741B7F99CE0A7AB49485597751B33432E57D1B395E0923EF0571D58823`
- pytest: `python -m pytest tests/test_elementary_math_nosolver2_vertical_add_sub_acquired.py -q`
- dependencies: `skill.math.count_successor.v1, skill.math.ten_within_add_sub.v1`
- allowed_inputs: bounded vertical-add cases within NoSolver2 evidence range
- abstain_policy: abstain if required column fact/borrow/carry process is unavailable or low-confidence
- evidence_report: `experiments/elementary_math_nosolver2_vertical_add_sub_acquired/APV21_ElementaryMath_NoSolver2_vertical_add_sub_acquired.md`

### skill.math.vertical_subtract.v1

- action_id: `action::skill.math.vertical_subtract.v1`
- title: Vertical subtraction process skill
- source_package: `experiments/elementary_math_nosolver2_vertical_add_sub_acquired/elementary_math_nosolver2_skill_package.json`
- package_sha256: `E89F04741B7F99CE0A7AB49485597751B33432E57D1B395E0923EF0571D58823`
- pytest: `python -m pytest tests/test_elementary_math_nosolver2_vertical_add_sub_acquired.py -q`
- dependencies: `skill.math.count_successor.v1, skill.math.ten_within_add_sub.v1`
- allowed_inputs: bounded vertical-subtract cases within NoSolver2 evidence range
- abstain_policy: abstain if borrow/subtract column process is unavailable or would go negative outside policy
- evidence_report: `experiments/elementary_math_nosolver2_vertical_add_sub_acquired/APV21_ElementaryMath_NoSolver2_vertical_add_sub_acquired.md`

### skill.math.vertical_multiply.v1

- action_id: `action::skill.math.vertical_multiply.v1`
- title: Vertical multiplication process skill
- source_package: `experiments/elementary_math_nosolver3_vertical_multiply_acquired/elementary_math_nosolver3_skill_package.json`
- package_sha256: `F4A3640AE409CB7B6564630E60241EF327A66599D3F7E2FE1A4E5A4366540F92`
- pytest: `python -m pytest tests/test_elementary_math_nosolver3_vertical_multiply_acquired.py -q`
- dependencies: `skill.math.count_successor.v1, skill.math.ten_within_add_sub.v1, skill.math.small_mul_div.v1, skill.math.vertical_add.v1`
- allowed_inputs: bounded vertical-multiply cases within NoSolver3 evidence range
- abstain_policy: abstain if partial-product or partial-sum process is unavailable
- evidence_report: `experiments/elementary_math_nosolver3_vertical_multiply_acquired/APV21_ElementaryMath_NoSolver3_vertical_multiply_acquired.md`

### skill.math.vertical_division_one_digit.v1

- action_id: `action::skill.math.vertical_division_one_digit.v1`
- title: One-digit divisor vertical division process skill
- source_package: `experiments/elementary_math_nosolver4_vertical_division_acquired/elementary_math_nosolver4_skill_package.json`
- package_sha256: `5A65FAF2228B90A531DA1A135FAFF55109A9657E465CB8294AF4AB0E81436487`
- pytest: `python -m pytest tests/test_elementary_math_nosolver4_vertical_division_acquired.py -q`
- dependencies: `skill.math.count_successor.v1, skill.math.ten_within_add_sub.v1, skill.math.vertical_subtract.v1, skill.math.vertical_multiply.v1`
- allowed_inputs: bounded one-digit divisor vertical division cases within NoSolver4 evidence range
- abstain_policy: abstain if trial quotient/multiply-back/subtract/remainder boundary is unavailable
- evidence_report: `experiments/elementary_math_nosolver4_vertical_division_acquired/APV21_ElementaryMath_NoSolver4_vertical_division_acquired.md`

### skill.math.repeated_subtract_division.v1

- action_id: `action::skill.math.repeated_subtract_division.v1`
- title: Repeated vertical-subtract division used for simple equation solving
- source_package: `experiments/math_fullchain1_pure_ap_equation_word_problem/math_fullchain1_skill_package.json`
- package_sha256: `4B978FC704AF720E7E7069BC942C7A73095C71B82E21C4B4E1B1C66AA38F345F`
- pytest: `python -m pytest tests/test_math_fullchain1_pure_ap_equation_word_problem.py -q`
- dependencies: `skill.math.count_successor.v1, skill.math.vertical_subtract.v1`
- allowed_inputs: exact small repeated-subtraction division cases seen in Math-FullChain-1 boundary
- abstain_policy: abstain when repeated subtraction cannot reach zero within learned guard/boundary
- evidence_report: `experiments/math_fullchain1_pure_ap_equation_word_problem/APV21_MathFullChain1_pure_ap_equation_word_problem.md`

### skill.math.linear_equation_word_problem.v1

- action_id: `action::skill.math.linear_equation_word_problem.v1`
- title: Simple same-groups plus extra linear equation word problem chain
- source_package: `experiments/math_fullchain1_pure_ap_equation_word_problem/math_fullchain1_skill_package.json`
- package_sha256: `4B978FC704AF720E7E7069BC942C7A73095C71B82E21C4B4E1B1C66AA38F345F`
- pytest: `python -m pytest tests/test_math_fullchain1_pure_ap_equation_word_problem.py -q`
- dependencies: `skill.math.vertical_subtract.v1, skill.math.repeated_subtract_division.v1, skill.math.vertical_multiply.v1, skill.math.vertical_add.v1`
- allowed_inputs: same-groups plus extra controlled word problems covered by Math-FullChain-1
- abstain_policy: abstain if template confidence, visible number roles, arithmetic subskill, or verification fails
- evidence_report: `experiments/math_fullchain1_pure_ap_equation_word_problem/APV21_MathFullChain1_pure_ap_equation_word_problem.md`

## 使用规则

- 证明底层技能本身时, 禁止调用该技能入口。
- 证明高层组合链路时, 可以调用已验证技能入口。
- 每次调用都必须保留 `evidence_ref` 和可展开 trace 策略。
- 输入超边界时必须 abstain。
