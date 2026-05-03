name: Feature Request
description: Suggest a new feature or improvement for NumCompute
title: "[FEATURE]: "
labels: ["enhancement"]

body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a feature!
        Help us improve **NumCompute** by describing your idea clearly.

  - type: textarea
    id: description
    attributes:
      label: Feature Description
      description: What feature would you like to see?
      placeholder: e.g. Add support for normalization in preprocessing module
    validations:
      required: true

  - type: textarea
    id: problem
    attributes:
      label: Problem / Use Case
      description: What problem does this feature solve?
      placeholder: e.g. Currently users cannot normalize data easily before training models

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How should this feature work?
      placeholder: e.g. Add a Normalizer class similar to StandardScaler

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Any alternative approaches you’ve thought about?
      placeholder: e.g. Using external libraries like sklearn (not ideal)

  - type: textarea
    id: context
    attributes:
      label: Additional Context
      description: Add screenshots, examples, or references
      placeholder: |
        Any extra information that helps explain your idea

  - type: dropdown
    id: contribute
    attributes:
      label: Would you like to work on this feature?
      options:
        - "Yes"
        - "No"

  - type: markdown
    attributes:
      value: |
        Thanks for your suggestion! 
        We will review it and get back to you soon.