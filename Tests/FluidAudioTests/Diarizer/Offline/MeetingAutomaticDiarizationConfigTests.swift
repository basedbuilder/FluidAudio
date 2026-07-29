import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class MeetingAutomaticDiarizationConfigTests: XCTestCase {

    func testDisconnectedSpeakerMaskSplittingIsDisabledByDefault() {
        let config = OfflineDiarizerConfig.default

        XCTAssertFalse(config.embedding.splitDisconnectedSpeakerMasks)
        XCTAssertFalse(config.splitsDisconnectedSpeakerMasksForAutomaticDiarization)
        XCTAssertFalse(config.consolidatesOverlappingAutomaticClusters)

        let sourceCompatibleEmbedding = OfflineDiarizerConfig.Embedding(
            batchSize: 32,
            excludeOverlap: true,
            minSegmentDurationSeconds: 1.0
        )
        XCTAssertFalse(sourceCompatibleEmbedding.splitDisconnectedSpeakerMasks)

        var automaticOptIn = config
        automaticOptIn.embedding.splitDisconnectedSpeakerMasks = true
        XCTAssertTrue(automaticOptIn.splitsDisconnectedSpeakerMasksForAutomaticDiarization)
        XCTAssertFalse(automaticOptIn.consolidatesOverlappingAutomaticClusters)

        automaticOptIn.clustering.preserveAutomaticAHCClusters = true
        XCTAssertTrue(automaticOptIn.consolidatesOverlappingAutomaticClusters)

        automaticOptIn.embedding.excludeOverlap = false
        XCTAssertFalse(automaticOptIn.consolidatesOverlappingAutomaticClusters)
        automaticOptIn.embedding.excludeOverlap = true

        automaticOptIn.clustering.numSpeakers = 2
        XCTAssertFalse(automaticOptIn.splitsDisconnectedSpeakerMasksForAutomaticDiarization)
        XCTAssertFalse(automaticOptIn.consolidatesOverlappingAutomaticClusters)
    }

    func testDisconnectedRunsProduceSeparateOverlapFreeEmbeddingMasks() {
        let masks = OfflineEmbeddingExtractor.disconnectedEmbeddingMasks(
            baseMask: [1, 1, 1, 0, 0, 1, 1],
            overlapFrames: [false, true, false, false, false, false, false],
            frameDuration: 0.1,
            minimumInactiveGapDuration: 0.19
        )

        XCTAssertEqual(masks.map(\.assignmentFrames), [0..<3, 5..<7])
        XCTAssertEqual(masks[0].embeddingWeights, [1, 0, 1, 0, 0, 0, 0])
        XCTAssertEqual(masks[1].embeddingWeights, [0, 0, 0, 0, 0, 1, 1])
        XCTAssertTrue(masks[0].assignmentFrames.contains(1))

        let overlapOnly = OfflineEmbeddingExtractor.disconnectedEmbeddingMasks(
            baseMask: [1, 1],
            overlapFrames: [true, true],
            frameDuration: 0.1,
            minimumInactiveGapDuration: 0.1
        )
        XCTAssertEqual(overlapOnly.map(\.assignmentFrames), [0..<2])
        XCTAssertEqual(overlapOnly[0].embeddingWeights, [0, 0])
    }

    func testDisconnectedRunsCoalesceAtGapThresholdAndSplitAboveIt() {
        let baseMask: [Float] = [1, 1, 0, 0, 1, 1]

        let belowThreshold = OfflineEmbeddingExtractor.disconnectedEmbeddingMasks(
            baseMask: baseMask,
            overlapFrames: [],
            frameDuration: 0.1,
            minimumInactiveGapDuration: 0.21
        )
        XCTAssertEqual(belowThreshold.map(\.assignmentFrames), [0..<6])

        let atThreshold = OfflineEmbeddingExtractor.disconnectedEmbeddingMasks(
            baseMask: baseMask,
            overlapFrames: [],
            frameDuration: 0.1,
            minimumInactiveGapDuration: 0.2
        )
        XCTAssertEqual(atThreshold.map(\.assignmentFrames), [0..<6])

        let aboveThreshold = OfflineEmbeddingExtractor.disconnectedEmbeddingMasks(
            baseMask: baseMask,
            overlapFrames: [],
            frameDuration: 0.1,
            minimumInactiveGapDuration: 0.19
        )
        XCTAssertEqual(aboveThreshold.map(\.assignmentFrames), [0..<2, 4..<6])
    }

    func testFrameLocalAssignmentsPreserveDisconnectedIdentityAndOverlap() {
        var config = OfflineDiarizerConfig(
            minSegmentDuration: 0,
            minGapDuration: 0.01,
            exclusiveSegments: false,
            segmentationMinDurationOn: 0,
            segmentationMinDurationOff: 0
        )
        config.embedding.splitDisconnectedSpeakerMasks = true

        let segmentation = SegmentationOutput(
            logProbs: [[[0]]],
            speakerWeights: [
                [
                    [1, 0],
                    [1, 0],
                    [0, 0],
                    [1, 1],
                    [1, 0],
                ]
            ],
            numChunks: 1,
            numFrames: 5,
            numSpeakers: 2,
            chunkOffsets: [0],
            frameDuration: 0.1
        )
        let timedEmbeddings = [
            makeTimedEmbedding(speakerIndex: 0, frames: 0...1),
            makeTimedEmbedding(speakerIndex: 0, frames: 3...4),
            makeTimedEmbedding(speakerIndex: 1, frames: 3...3),
        ]
        let frameLocalAssignments = OfflineReconstruction.buildFrameLocalClusterAssignments(
            segmentation: segmentation,
            timedEmbeddings: timedEmbeddings,
            assignments: [0, 1, 0],
            clusterCount: 2
        )

        XCTAssertEqual(
            frameLocalAssignments[0],
            [
                OfflineFrameLocalClusterAssignment(
                    speakerIndex: 0,
                    frames: 0...1,
                    cluster: 0
                ),
                OfflineFrameLocalClusterAssignment(
                    speakerIndex: 0,
                    frames: 3...4,
                    cluster: 1
                ),
                OfflineFrameLocalClusterAssignment(
                    speakerIndex: 1,
                    frames: 3...3,
                    cluster: 0
                ),
            ]
        )

        let reconstruction = OfflineReconstruction(config: config)
        let legacySegments = reconstruction.buildSegments(
            segmentation: segmentation,
            hardClusters: [[1, 0]],
            centroids: [[1, 0], [0, 1]]
        )
        XCTAssertTrue(
            legacySegments.contains {
                $0.speakerId == "S2"
                    && abs($0.startTimeSeconds - 0) < 1e-6
                    && abs($0.endTimeSeconds - 0.2) < 1e-6
            }
        )

        let segments = reconstruction.buildSegments(
            segmentation: segmentation,
            hardClusters: [[1, 0]],
            centroids: [[1, 0], [0, 1]],
            frameLocalClusters: frameLocalAssignments
        )

        XCTAssertTrue(
            segments.contains {
                $0.speakerId == "S1"
                    && abs($0.startTimeSeconds - 0) < 1e-6
                    && abs($0.endTimeSeconds - 0.2) < 1e-6
            }
        )
        XCTAssertTrue(
            segments.contains {
                $0.speakerId == "S1"
                    && abs($0.startTimeSeconds - 0.3) < 1e-6
                    && abs($0.endTimeSeconds - 0.4) < 1e-6
            }
        )
        XCTAssertTrue(
            segments.contains {
                $0.speakerId == "S2"
                    && abs($0.startTimeSeconds - 0.3) < 1e-6
                    && abs($0.endTimeSeconds - 0.5) < 1e-6
            }
        )

        let publicEmbeddings = OfflineDiarizerManager.buildPublicChunkEmbeddings(
            timedEmbeddings: timedEmbeddings,
            assignments: [0, 1, 0],
            logger: AppLogger(category: "MeetingAutomaticDiarizationConfigTests")
        )
        XCTAssertEqual(publicEmbeddings.prefix(2).map(\.chunkIndex), [0, 0])
        XCTAssertEqual(publicEmbeddings.prefix(2).map(\.speakerIndex), [0, 0])
        XCTAssertEqual(publicEmbeddings[0].startTimeSeconds, 0, accuracy: 1e-9)
        XCTAssertEqual(publicEmbeddings[1].startTimeSeconds, 0.3, accuracy: 1e-9)
    }

    func testFrameLocalReconstructionDoesNotInventMissingOverlapIdentity() {
        var config = OfflineDiarizerConfig(
            minSegmentDuration: 0,
            minGapDuration: 0.01,
            exclusiveSegments: false,
            segmentationMinDurationOn: 0,
            segmentationMinDurationOff: 0
        )
        config.embedding.splitDisconnectedSpeakerMasks = true

        let segmentation = SegmentationOutput(
            logProbs: [[[0]]],
            speakerWeights: [
                [
                    [1, 1],
                    [1, 1],
                ]
            ],
            numChunks: 1,
            numFrames: 2,
            numSpeakers: 2,
            chunkOffsets: [0],
            frameDuration: 0.1
        )
        let timedEmbeddings = [
            makeTimedEmbedding(speakerIndex: 0, frames: 0...1)
        ]
        let frameLocalAssignments =
            OfflineReconstruction.buildFrameLocalClusterAssignments(
                segmentation: segmentation,
                timedEmbeddings: timedEmbeddings,
                assignments: [0],
                clusterCount: 2
            )

        let segments = OfflineReconstruction(config: config).buildSegments(
            segmentation: segmentation,
            hardClusters: [[0, 1]],
            centroids: [[1, 0], [0, 1]],
            frameLocalClusters: frameLocalAssignments
        )

        XCTAssertEqual(Set(segments.map(\.speakerId)), ["S1"])
    }

    func testMinimumActiveRatioIncludesEqualityAtThreshold() {
        XCTAssertTrue(
            OfflineEmbeddingExtractor.meetsMinimumActiveRatio(
                cleanActivity: 20,
                frameCount: 100,
                minimumActiveRatio: 0.2
            )
        )
        XCTAssertFalse(
            OfflineEmbeddingExtractor.meetsMinimumActiveRatio(
                cleanActivity: 19.99,
                frameCount: 100,
                minimumActiveRatio: 0.2
            )
        )
    }

    func testDefaultUnconstrainedConfigurationUsesVBxCentroids() {
        let manager = OfflineDiarizerManager(config: .default)

        XCTAssertEqual(
            manager.computeCentroids(
                trainingEmbeddings: trainingEmbeddings,
                vbxOutput: vbxOutput,
                initialClusters: initialClusters
            ).centroids,
            vbxCentroids
        )
    }

    func testOptInUnconstrainedConfigurationUsesInitialAHCClusters() {
        var config = OfflineDiarizerConfig.default
        config.clustering.preserveAutomaticAHCClusters = true
        let manager = OfflineDiarizerManager(config: config)

        XCTAssertEqual(
            manager.computeCentroids(
                trainingEmbeddings: trainingEmbeddings,
                vbxOutput: vbxOutput,
                initialClusters: initialClusters
            ).centroids,
            ahcCentroids
        )
    }

    func testOptInConstrainedConfigurationsRetainVBxCentroids() {
        for config in constrainedConfigurations() {
            let manager = OfflineDiarizerManager(config: config)

            XCTAssertEqual(
                manager.computeCentroids(
                    trainingEmbeddings: trainingEmbeddings,
                    vbxOutput: vbxOutput,
                    initialClusters: initialClusters
                ).centroids,
                vbxCentroids
            )
        }
    }

    func testOptInConstrainedConfigurationsRetainAdjustedKMeansCentroids() {
        let adjustedCentroids = [[101.0, 0], [202.0, 0]]
        let adjustedOutput = VBxOutput(
            gamma: vbxOutput.gamma,
            pi: vbxOutput.pi,
            hardClusters: vbxOutput.hardClusters,
            centroids: adjustedCentroids,
            numClusters: adjustedCentroids.count,
            elbos: [],
            wasAdjusted: true,
            originalClusterCount: 3
        )

        for config in constrainedConfigurations() {
            let manager = OfflineDiarizerManager(config: config)

            XCTAssertEqual(
                manager.computeCentroids(
                    trainingEmbeddings: trainingEmbeddings,
                    vbxOutput: adjustedOutput,
                    initialClusters: initialClusters
                ).centroids,
                adjustedCentroids
            )
        }
    }

    private let trainingEmbeddings = [
        [0.0, 0],
        [2.0, 0],
        [10.0, 0],
        [12.0, 0],
    ]
    private let initialClusters = [0, 0, 1, 1]
    private let vbxCentroids = [[5.0, 0], [7.0, 0]]
    private let ahcCentroids = [[1.0, 0], [11.0, 0]]

    private var vbxOutput: VBxOutput {
        VBxOutput(
            gamma: [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            pi: [0.5, 0.5],
            hardClusters: [[0, 1, 0, 1]],
            centroids: [],
            numClusters: 2,
            elbos: []
        )
    }

    private func constrainedConfigurations() -> [OfflineDiarizerConfig] {
        [
            makeConfig(numSpeakers: 2),
            makeConfig(minSpeakers: 2),
            makeConfig(maxSpeakers: 4),
            makeConfig(minSpeakers: 2, maxSpeakers: 4),
        ]
    }

    private func makeConfig(
        minSpeakers: Int? = nil,
        maxSpeakers: Int? = nil,
        numSpeakers: Int? = nil
    ) -> OfflineDiarizerConfig {
        var config = OfflineDiarizerConfig.default
        config.clustering.minSpeakers = minSpeakers
        config.clustering.maxSpeakers = maxSpeakers
        config.clustering.numSpeakers = numSpeakers
        config.clustering.preserveAutomaticAHCClusters = true
        return config
    }

    private func makeTimedEmbedding(
        speakerIndex: Int,
        frames: ClosedRange<Int>
    ) -> TimedEmbedding {
        TimedEmbedding(
            chunkIndex: 0,
            speakerIndex: speakerIndex,
            startFrame: frames.lowerBound,
            endFrame: frames.upperBound,
            frameWeights: [],
            startTime: Double(frames.lowerBound) * 0.1,
            endTime: Double(frames.upperBound + 1) * 0.1,
            embedding256: [1, 0],
            rho128: [1, 0]
        )
    }
}
