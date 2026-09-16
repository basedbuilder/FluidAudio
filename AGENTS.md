# FluidAudio - Agent Development Guide

## Build & Test Commands

```bash
swift build                                    # Build project
swift build -c release                        # Release build
swift test                                     # Run all tests
swift test --filter CITests                   # Run single test class
swift test --filter CITests.testPackageImports # Run single test method
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/
```

## Architecture

- **FluidAudio/**: Main library (ASR/, Diarizer/, VAD/, Shared/ modules)
- **FluidAudioCLI/**: CLI tool with benchmarking and processing commands
- **Tests/FluidAudioTests/**: Comprehensive test suite
- **Models**: Auto-downloaded from HuggingFace with CoreML compilation
- **Processing Pipeline**: Audio → VAD → Diarization → ASR → Timestamped transcripts

## Critical Rules

- **NEVER** use `@unchecked Sendable` - implement proper thread safety with actors/MainActor
- **NEVER** create dummy/mock models or synthetic audio data - use real models only
- Complete the requested behavior with the smallest maintainable solution; ask only when a missing choice materially changes that behavior.
- **NEVER** run `git push` unless explicitly requested by user
- Add regression coverage when it proves changed behavior or a meaningful failure mode.

## Code Style (swift-format config)

- Line length: 120 chars, 4-space indentation
- Import order: Alphabetical preferred (`import CoreML`, `import Foundation`, `import OSLog`), but OrderedImports rule is disabled due to Swift 6.1 (GitHub Actions CI) vs 6.3 (local) formatter incompatibility
- Naming: lowerCamelCase for variables/functions, UpperCamelCase for types
- Error handling: Use proper Swift error handling, no force unwrapping in production
- Documentation: Triple-slash comments (`///`) for public APIs
- Thread safety: Use actors, `@MainActor`, or proper locking - never `@unchecked Sendable`
- Control flow: Prefer flattened if statements with early returns/continues over nested if statements. Use guard statements and inverted conditions to exit early. Nested if statements should be absolutely avoided.

## Clean code

- When adding new interfaces, make sure that the API is consistent with the other model managers
- Files should be isolated and the code should contain a single responsibility for each

## Mobius Plan

Use a saved plan when requested or when architecture, migration, recovery or cross-session coordination needs it. Read applicable `PLANS.md` policy when creating or implementing one; routine bounded fixes can use an in-thread plan. Keep this repo's plans in `.mobius/` and out of GitHub.
