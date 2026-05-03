name: Bug Report
description: Report a bug to help improve NumCompute
title: "[BUG]: "
labels: ["bug"]

body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting a bug! 
        Please fill out the details below so we can fix it quickly.

  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: Clearly describe what the bug is.
      placeholder: e.g. StandardScaler returns incorrect values when input has constant features
    validations:
      required: true

  - type: textarea
    id: steps
    attributes:
      label: 🔁 Steps to Reproduce
      description: Steps to reproduce the issue
      placeholder: |
        1. Import StandardScaler
        2. Pass input [5,5,5]
        3. Call transform()
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: ✅ Expected Behavior
      description: What should have happened?
      placeholder: Output should be [0, 0, 0]

  - type: textarea
    id: actual
    attributes:
      label: ❌ Actual Behavior
      description: What actually happened?
      placeholder: Output is [NaN, NaN, NaN]

  - type: textarea
    id: logs
    attributes:
      label: Logs / Screenshots
      description: Paste error logs or attach screenshots
      placeholder: Paste traceback or error message here

  - type: input
    id: python_version
    attributes:
      label: Python Version
      placeholder: e.g. 3.10

  - type: input
    id: os
    attributes:
      label: 💻 Operating System
      placeholder: e.g. macOS / Windows

  - type: input
    id: package_version
    attributes:
      label: NumCompute Version
      placeholder: e.g. 0.1.2

  - type: dropdown
    id: contribute
    attributes:
      label: 🤝 Would you like to work on this issue?
      options:
        - "Yes"
        - "No"