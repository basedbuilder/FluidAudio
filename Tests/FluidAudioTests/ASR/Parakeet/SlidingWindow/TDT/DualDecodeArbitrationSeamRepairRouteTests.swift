import Foundation
import XCTest

final class DualDecodeArbitrationSeamRepairRouteTests: XCTestCase {

    func testDualDecodeRepairsAfterMergeBeforeProjectionWithExistingInputs() throws {
        let testFile = URL(fileURLWithPath: #filePath)
        let repositoryRoot =
            testFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/DualDecodeArbitration.swift"
        )
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let callerURL = repositoryRoot.appendingPathComponent(
            "Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/ChunkProcessor.swift"
        )
        let callerSource = try String(contentsOf: callerURL, encoding: .utf8)

        let functionStart = try XCTUnwrap(
            source.range(of: "func processWithDualDecodeArbitration(")
        )
        let functionSource = source[functionStart.lowerBound...]
        XCTAssertNotNil(
            functionSource.range(of: "speechRmsThresholdProvider: () throws -> Float")
        )
        XCTAssertNotNil(
            callerSource.range(
                of: "speechRmsThresholdProvider: { try adaptiveSpeechRmsThreshold() }"
            )
        )
        let collapse = try XCTUnwrap(
            functionSource.range(of: "mergedTokens = collapseSeamWordDuplicates(")
        )
        let repairGuard = try XCTUnwrap(
            functionSource.range(
                of: "if chunkOutputs.count > 1, mergedTokens.count > 1, await manager.seamGapRepair"
            )
        )
        let threshold = try XCTUnwrap(
            functionSource.range(of: "let speechRmsThreshold = try speechRmsThresholdProvider()")
        )
        let repair = try XCTUnwrap(
            functionSource.range(of: "mergedTokens = try await repairSeamGaps(")
        )
        let worker = try XCTUnwrap(
            functionSource.range(of: "using: worker", range: repair.lowerBound..<functionSource.endIndex)
        )
        let projection = try XCTUnwrap(
            functionSource.range(of: "let allTokens = mergedTokens.map")
        )

        XCTAssertLessThan(collapse.lowerBound, repairGuard.lowerBound)
        XCTAssertLessThan(repairGuard.lowerBound, threshold.lowerBound)
        XCTAssertLessThan(threshold.lowerBound, repair.lowerBound)
        XCTAssertLessThan(repair.lowerBound, worker.lowerBound)
        XCTAssertLessThan(worker.lowerBound, projection.lowerBound)
    }
}
