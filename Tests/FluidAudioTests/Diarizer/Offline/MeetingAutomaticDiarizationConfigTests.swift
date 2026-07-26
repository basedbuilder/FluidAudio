import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class MeetingAutomaticDiarizationConfigTests: XCTestCase {

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
}
