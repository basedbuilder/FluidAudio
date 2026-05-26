import XCTest

@testable import FluidAudio

final class CustomVocabularyDebugSettingsTests: XCTestCase {

    func testVerboseLoggingDefaultsToFalse() {
        XCTAssertFalse(CustomVocabularyDebugSettings.verboseLoggingEnabled(environment: [:]))
    }

    func testVerboseLoggingAcceptsTruthyValues() {
        for value in ["1", "true", "TRUE", " yes ", "On"] {
            XCTAssertTrue(
                CustomVocabularyDebugSettings.verboseLoggingEnabled(
                    environment: [CustomVocabularyDebugSettings.environmentVariable: value]
                ),
                "\(value) should enable custom vocabulary debug logging"
            )
        }
    }

    func testVerboseLoggingRejectsFalsyAndUnknownValues() {
        for value in ["", "0", "false", "no", "off", "debug"] {
            XCTAssertFalse(
                CustomVocabularyDebugSettings.verboseLoggingEnabled(
                    environment: [CustomVocabularyDebugSettings.environmentVariable: value]
                ),
                "\(value) should not enable custom vocabulary debug logging"
            )
        }
    }

    func testRescorerExplicitDebugModeOverridesEnvironment() async throws {
        let targetDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("CustomVocabularyDebugSettingsTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: targetDir) }

        _ = try await CtcModels.downloadTokenizerAssets(to: targetDir, variant: .ctc110m)
        let rescorer = try await VocabularyRescorer.create(
            vocabulary: CustomVocabularyContext(terms: []),
            ctcModelDirectory: targetDir,
            debugMode: false
        )

        XCTAssertFalse(rescorer.debugMode)
    }
}
