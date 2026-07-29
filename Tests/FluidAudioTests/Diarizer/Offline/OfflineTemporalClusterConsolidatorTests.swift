import XCTest

@testable import FluidAudio

final class OfflineTemporalClusterConsolidatorTests: XCTestCase {
    func testDuplicateCleanObservationsAcrossChunksMerge() {
        let segmentation = makeSegmentation(
            offsets: [0, 2],
            frameDuration: 1,
            weights: [
                repeatedFrames([1], count: 5),
                repeatedFrames([1], count: 5),
            ]
        )
        let embeddings = [
            makeEmbedding(
                chunk: 0,
                speaker: 0,
                frames: 2...4,
                weights: [0, 0, 1, 1, 1]
            ),
            makeEmbedding(
                chunk: 1,
                speaker: 0,
                frames: 0...2,
                weights: [1, 1, 1, 0, 0]
            ),
        ]

        let result = OfflineTemporalClusterConsolidator.consolidate(
            timedEmbeddings: embeddings,
            initialClusters: [0, 1],
            segmentation: segmentation
        )

        XCTAssertEqual(result, [0, 0])
    }

    func testSequentialRunsWithoutSharedCleanFramesRemainSeparate() {
        let segmentation = makeSegmentation(
            offsets: [0, 4],
            frameDuration: 1,
            weights: [
                repeatedFrames([1], count: 3),
                repeatedFrames([1], count: 3),
            ]
        )
        let embeddings = [
            makeEmbedding(
                chunk: 0,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1, 0]
            ),
            makeEmbedding(
                chunk: 1,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1, 0]
            ),
        ]

        let result = OfflineTemporalClusterConsolidator.consolidate(
            timedEmbeddings: embeddings,
            initialClusters: [0, 1],
            segmentation: segmentation
        )

        XCTAssertEqual(result, [0, 1])
    }

    func testOneSharedBoundaryFrameDoesNotMergeSequentialSpeakers() {
        let segmentation = makeSegmentation(
            offsets: [0, 0],
            frameDuration: 1,
            weights: [
                repeatedFrames([1], count: 3),
                repeatedFrames([1], count: 3),
            ]
        )
        let embeddings = [
            makeEmbedding(
                chunk: 0,
                speaker: 0,
                frames: 1...1,
                weights: [0, 1, 0]
            ),
            makeEmbedding(
                chunk: 1,
                speaker: 0,
                frames: 1...1,
                weights: [0, 1, 0]
            ),
        ]

        let result = OfflineTemporalClusterConsolidator.consolidate(
            timedEmbeddings: embeddings,
            initialClusters: [0, 1],
            segmentation: segmentation
        )

        XCTAssertEqual(result, [0, 1])
    }

    func testSameChunkCoactivityVetoesTemporalMustLink() {
        let segmentation = makeSegmentation(
            offsets: [0, 0],
            frameDuration: 1,
            weights: [
                [
                    [1, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1],
                    [0, 1],
                ],
                repeatedFrames([1, 0], count: 5),
            ]
        )
        let embeddings = [
            makeEmbedding(
                chunk: 0,
                speaker: 0,
                frames: 0...2,
                weights: [1, 1, 0, 0, 0]
            ),
            makeEmbedding(
                chunk: 0,
                speaker: 1,
                frames: 2...4,
                weights: [0, 0, 0, 1, 1]
            ),
            makeEmbedding(
                chunk: 1,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1, 0, 0, 0]
            ),
        ]

        let result = OfflineTemporalClusterConsolidator.consolidate(
            timedEmbeddings: embeddings,
            initialClusters: [0, 1, 1],
            segmentation: segmentation
        )

        XCTAssertEqual(result, [0, 1, 1])
    }

    func testTransitiveMustLinkCannotBypassComponentConflict() {
        let segmentation = makeSegmentation(
            offsets: [0, 0, 10, 10, 20],
            frameDuration: 1,
            weights: [
                repeatedFrames([1, 0], count: 3),
                repeatedFrames([1, 0], count: 3),
                repeatedFrames([1, 0], count: 3),
                repeatedFrames([1, 0], count: 3),
                [[1, 1], [0, 0], [0, 0]],
            ]
        )
        let embeddings = [
            makeEmbedding(
                chunk: 0,
                speaker: 0,
                frames: 0...2,
                weights: [1, 1, 1]
            ),
            makeEmbedding(
                chunk: 1,
                speaker: 0,
                frames: 0...2,
                weights: [1, 1, 1]
            ),
            makeEmbedding(
                chunk: 2,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1, 0]
            ),
            makeEmbedding(
                chunk: 3,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1, 0]
            ),
            makeEmbedding(
                chunk: 4,
                speaker: 0,
                frames: 0...0,
                weights: [0, 0, 0]
            ),
            makeEmbedding(
                chunk: 4,
                speaker: 1,
                frames: 0...0,
                weights: [0, 0, 0]
            ),
        ]

        let result = OfflineTemporalClusterConsolidator.consolidate(
            timedEmbeddings: embeddings,
            initialClusters: [0, 1, 1, 2, 0, 2],
            segmentation: segmentation
        )

        XCTAssertEqual(result[0], result[1])
        XCTAssertEqual(result[1], result[2])
        XCTAssertNotEqual(result[0], result[3])
        XCTAssertEqual(result[0], result[4])
        XCTAssertEqual(result[3], result[5])
        XCTAssertEqual(Set(result).count, 2)
    }

    func testMalformedAndOneChunkInputsKeepIdentity() {
        let oneChunk = makeSegmentation(
            offsets: [0],
            frameDuration: 1,
            weights: [repeatedFrames([1], count: 2)]
        )
        let embedding = makeEmbedding(
            chunk: 0,
            speaker: 0,
            frames: 0...1,
            weights: [1, 1]
        )

        XCTAssertEqual(
            OfflineTemporalClusterConsolidator.consolidate(
                timedEmbeddings: [],
                initialClusters: [],
                segmentation: oneChunk
            ),
            []
        )
        XCTAssertEqual(
            OfflineTemporalClusterConsolidator.consolidate(
                timedEmbeddings: [embedding],
                initialClusters: [4, 5],
                segmentation: oneChunk
            ),
            [4, 5]
        )
        XCTAssertEqual(
            OfflineTemporalClusterConsolidator.consolidate(
                timedEmbeddings: [embedding],
                initialClusters: [4],
                segmentation: oneChunk
            ),
            [4]
        )
    }

    func testAcceptedMergeRemapsDeterministicallyAndContiguously() {
        let segmentation = makeSegmentation(
            offsets: [0, 0, 10],
            frameDuration: 1,
            weights: [
                repeatedFrames([1], count: 2),
                repeatedFrames([1], count: 2),
                repeatedFrames([1], count: 2),
            ]
        )
        let embeddings = [
            makeEmbedding(
                chunk: 0,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1]
            ),
            makeEmbedding(
                chunk: 1,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1]
            ),
            makeEmbedding(
                chunk: 2,
                speaker: 0,
                frames: 0...1,
                weights: [1, 1]
            ),
        ]

        let first = OfflineTemporalClusterConsolidator.consolidate(
            timedEmbeddings: embeddings,
            initialClusters: [5, 2, 9],
            segmentation: segmentation
        )
        let second = OfflineTemporalClusterConsolidator.consolidate(
            timedEmbeddings: embeddings,
            initialClusters: [5, 2, 9],
            segmentation: segmentation
        )

        XCTAssertEqual(first, [0, 0, 1])
        XCTAssertEqual(second, first)
    }

    private func makeSegmentation(
        offsets: [Double],
        frameDuration: Double,
        weights: [[[Float]]]
    ) -> SegmentationOutput {
        SegmentationOutput(
            logProbs: [],
            speakerWeights: weights,
            numChunks: offsets.count,
            numFrames: weights.map(\.count).max() ?? 0,
            numSpeakers:
                weights
                .flatMap { $0 }
                .map(\.count)
                .max() ?? 0,
            chunkOffsets: offsets,
            frameDuration: frameDuration
        )
    }

    private func repeatedFrames(
        _ weights: [Float],
        count: Int
    ) -> [[Float]] {
        Array(repeating: weights, count: count)
    }

    private func makeEmbedding(
        chunk: Int,
        speaker: Int,
        frames: ClosedRange<Int>,
        weights: [Float]
    ) -> TimedEmbedding {
        TimedEmbedding(
            chunkIndex: chunk,
            speakerIndex: speaker,
            startFrame: frames.lowerBound,
            endFrame: frames.upperBound,
            frameWeights: weights,
            startTime: Double(frames.lowerBound),
            endTime: Double(frames.upperBound + 1),
            embedding256: [1, 0],
            rho128: [1, 0]
        )
    }
}
