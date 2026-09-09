import XCTest
@testable import FluidAudio

final class SpeechComponentEvidenceTests: XCTestCase {
    func testEmptyNearestPoolCannotCreateBiasOnlySpeaker() {
        XCTAssertNil(OfflineSpeechComponentExtractor.cleanPoolingWeights([0, 1, 1, 0], featureFrames: 8))
        XCTAssertEqual(OfflineSpeechComponentExtractor.cleanPoolingWeights([1, 0, 0, 1], featureFrames: 16), [1, 0])
        XCTAssertEqual(OfflineSpeechComponentExtractor.cleanPoolingWeights([0, 1, 1, 0], featureFrames: 16), [0, 1])
    }

    func testUnsupportedVBxMergeKeepsDistinctSpeakers() throws {
        let result = try refine(intervals: [[0..<100], [200..<300]])
        XCTAssertEqual(result, [[1, 0], [0, 1]])
    }

    func testSharedCleanAudioSupportsMergeAtTrainingFloor() throws {
        let result = try refine(intervals: [[0..<100], [50..<150]])
        XCTAssertEqual(result, [[0.5, 0.5]])
        XCTAssertEqual(try refine(intervals: [[0..<100], [51..<151]]), [[1, 0], [0, 1]])
    }

    func testAcousticSupportPreservesOriginalVBxCentroid() throws {
        let result = try OfflineSpeakerMergeSupport.refine(
            centroids: [[0.95, 0.05]], retainedColumns: [0], gamma: [[1], [1]],
            initialClusters: [0, 1], trainingEmbeddings: [[1, 0], [0.9, 0.1]],
            cleanIntervals: [[0..<100], [200..<300]], minimumSharedSamples: 50
        )
        XCTAssertEqual(result, [[0.95, 0.05]])
    }

    func testMissingPhysicalEvidenceCannotPretendToValidateMerge() {
        XCTAssertThrowsError(try refine(intervals: [[0..<100], []]))
    }

    private func refine(intervals: [[Range<Int>]]) throws -> [[Double]] {
        try OfflineSpeakerMergeSupport.refine(
            centroids: [[0.5, 0.5]], retainedColumns: [0], gamma: [[1], [1]],
            initialClusters: [0, 1], trainingEmbeddings: [[1, 0], [0, 1]],
            cleanIntervals: intervals, minimumSharedSamples: 50
        )
    }
}
