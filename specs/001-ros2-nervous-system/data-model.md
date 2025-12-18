# Data Model: ROS 2 Nervous System Module

## Content Entities

### Module
- **name**: String (required) - The module name (e.g., "ROS 2 Nervous System")
- **description**: String (required) - Brief description of the module
- **chapters**: Array<Chapter> (required) - List of chapters in the module
- **target_audience**: String (required) - Intended audience (e.g., "AI and software engineering students")
- **prerequisites**: Array<String> - Knowledge required before starting the module

### Chapter
- **id**: String (required) - Unique identifier for the chapter
- **title**: String (required) - Display title of the chapter
- **content**: String (required) - Markdown content of the chapter
- **learning_objectives**: Array<String> (required) - What students will learn
- **prerequisites**: Array<String> - Knowledge required for this chapter
- **duration**: Number (optional) - Estimated completion time in minutes
- **related_topics**: Array<String> - Cross-references to other content

## Content Relationships

```
Module 1 -- * Chapter
```

A module contains multiple chapters that build upon each other in a sequential learning path.

## Validation Rules

- Module name must be 3-100 characters
- Module description must be 10-500 characters
- Each module must have 1-10 chapters
- Each chapter title must be 5-100 characters
- Each chapter must have at least one learning objective
- Chapter content must be valid Markdown
- Chapter duration must be between 15-240 minutes if specified

## State Transitions

Since this is static documentation content, there are no state transitions. Content is created, reviewed, and published as static Markdown files.