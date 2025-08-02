# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the dynamic-moe-router-kit project.

## Format

We use the format described by Michael Nygard in his article "[Documenting Architecture Decisions](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions)".

Each ADR has the following sections:

- **Title**: Short descriptive title
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What is the issue that we're seeing that is motivating this decision?
- **Decision**: What is the change that we're proposing and/or doing?
- **Consequences**: What becomes easier or more difficult to do because of this change?

## Index

- [ADR-001: Multi-Backend Architecture](001-multi-backend-architecture.md)
- [ADR-002: Dynamic Routing Algorithm](002-dynamic-routing-algorithm.md)
- [ADR-003: Complexity Estimation Strategy](003-complexity-estimation-strategy.md)

## Creating New ADRs

1. Copy the template from `000-template.md`
2. Number it sequentially
3. Fill in all sections
4. Update this index
5. Submit for review