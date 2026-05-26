import XCTest

@testable import FluidAudio

final class CtcTokenizerAssetTests: XCTestCase {

    func testTokenizerAssetsDownloadWithoutAcousticModels() async throws {
        let targetDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("CtcTokenizerAssetTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: targetDir) }

        let downloadedDir = try await CtcModels.downloadTokenizerAssets(to: targetDir, variant: .ctc110m)
        XCTAssertEqual(downloadedDir, targetDir)

        for assetName in CtcModels.tokenizerAssetNames {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: targetDir.appendingPathComponent(assetName).path),
                "\(assetName) should be downloaded"
            )
        }

        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: targetDir.appendingPathComponent(ModelNames.CTC.melSpectrogramPath).path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: targetDir.appendingPathComponent(ModelNames.CTC.audioEncoderPath).path
            )
        )

        let metadata = try CtcModels.loadVocabularyMetadata(from: targetDir)
        XCTAssertEqual(metadata.directory, targetDir)
        XCTAssertEqual(metadata.vocabularyCount, 1024)
        XCTAssertEqual(metadata.blankId, 1024)

        let tokenizer = try await CtcTokenizer.load(from: targetDir)
        XCTAssertFalse(tokenizer.encode("Eric").isEmpty)
    }

    func testCtcHeadPreflightOnlyRunsForPlausibleCandidates() async throws {
        let targetDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("CtcTokenizerAssetTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: targetDir) }

        _ = try await CtcModels.downloadTokenizerAssets(to: targetDir, variant: .ctc110m)
        let tokenizer = try await CtcTokenizer.load(from: targetDir)
        let vocabulary = CustomVocabularyContext(
            terms: [
                CustomVocabularyTerm(text: "Eric", ctcTokenIds: tokenizer.encode("Eric"))
            ]
        )
        let rescorer = try await VocabularyRescorer.create(
            vocabulary: vocabulary,
            ctcModelDirectory: targetDir
        )

        XCTAssertFalse(
            rescorer.hasPotentialCtcTokenRescoreCandidate(
                transcript: "Let's now test this with dictionary.",
                tokenTimings: tokenTimings(["Let's", " now", " test", " this", " with", " dictionary"])
            )
        )
        XCTAssertFalse(
            rescorer.hasPotentialCtcTokenRescoreCandidate(
                transcript: "Hello Eric.",
                tokenTimings: tokenTimings(["Hello", " Eric"])
            )
        )
        XCTAssertTrue(
            rescorer.hasPotentialCtcTokenRescoreCandidate(
                transcript: "Hello Herik.",
                tokenTimings: tokenTimings(["Hello", " Herik"])
            )
        )
    }

    private func tokenTimings(_ tokens: [String]) -> [TokenTiming] {
        tokens.enumerated().map { index, token in
            TokenTiming(
                token: token,
                tokenId: index,
                startTime: Double(index) * 0.25,
                endTime: Double(index + 1) * 0.25,
                confidence: 0.9
            )
        }
    }
}
