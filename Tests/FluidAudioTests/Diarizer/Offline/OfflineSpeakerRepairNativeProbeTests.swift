import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Opt-in diagnostic over frozen real model outputs. It never downloads models.
@available(macOS 14.0, iOS 17.0, *)
final class OfflineSpeakerRepairNativeProbeTests: XCTestCase {
    private struct WaveformManifest: Decodable {
        struct Case: Decodable {
            let fixtureID: String
            let audio: String
            let expectedEmbeddingCount: Int
        }
        let cases: [Case]
        let modelsDirectory: String
        let embeddingPackage: String
        let phi: [Double]
    }

    func testCompleteWaveformPipeline() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let inputPath = environment["PARASPEECH_SPEAKER_WAVEFORM_INPUT"],
            let outputPath = environment["PARASPEECH_SPEAKER_WAVEFORM_OUTPUT"]
        else {
            throw XCTSkip("Explicit frozen waveform corpus paths are required")
        }
        let manifest = try JSONDecoder().decode(
            WaveformManifest.self, from: Data(contentsOf: URL(fileURLWithPath: inputPath)))
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = .cpuOnly
        let directory = URL(fileURLWithPath: manifest.modelsDirectory, isDirectory: true)
        func load(_ name: String) throws -> MLModel {
            try MLModel(
                contentsOf: directory.appendingPathComponent(name + ".mlmodelc"), configuration: modelConfiguration)
        }
        let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: manifest.embeddingPackage))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let models = try OfflineDiarizerModels(
            segmentationModel: load("Segmentation"), fbankModel: load("FBank"),
            embeddingModel: load("Embedding"), pldaRhoModel: load("PldaRho"),
            pldaPsi: manifest.phi, compilationDuration: 0,
            speechComponentEmbeddingModel: MLModel(contentsOf: compiled, configuration: modelConfiguration)
        )
        var configuration = OfflineDiarizerConfig(clusteringThreshold: 0.75)
        configuration.segmentationStepRatio = 0.1
        configuration.useSpeechComponentEmbeddings = true
        let manager = OfflineDiarizerManager(config: configuration)
        manager.initialize(models: models)
        for entry in manifest.cases {
            let start = Date()
            let (audio, duration) = try AudioSourceFactory().makeDiskBackedSource(
                from: URL(fileURLWithPath: entry.audio), targetSampleRate: 16_000)
            defer { audio.cleanup() }
            let prepared = try await manager.prepare(audioSource: audio, audioLoadingSeconds: duration)
            XCTAssertEqual(prepared.embeddingCount, entry.expectedEmbeddingCount, entry.fixtureID)
            let result = try manager.cluster(prepared)
            func segments(_ values: [TimedSpeakerSegment]) -> [[String: Any]] {
                values.map {
                    [
                        "speakerID": $0.speakerId, "startTime": $0.startTimeSeconds, "endTime": $0.endTimeSeconds,
                        "qualityScore": $0.qualityScore,
                    ]
                }
            }
            let report: [String: Any] = [
                "status": "completed", "kind": "shipping_rule_component_arithmetic_control",
                "fixtureID": entry.fixtureID,
                "speakerCount": Set(result.speakerActivitySegments.map(\.speakerId)).count,
                "segments": segments(result.segments), "activitySegments": segments(result.speakerActivitySegments),
                "nativeSwiftWaveformPipelineExecuted": true, "appIntegrationExecuted": false,
                "embeddingCount": prepared.embeddingCount,
                "trainingCount": prepared.componentTraining?.indices.count ?? 0,
                "timedRows": prepared.timedEmbeddings.map {
                    [
                        "chunk": $0.chunkIndex, "slot": $0.speakerIndex, "firstFrame": $0.startFrame,
                        "lastFrameExclusive": $0.endFrame,
                    ]
                },
                "components": (0..<prepared.segmentation.numChunks).flatMap { chunk in
                    (0..<prepared.segmentation.numSpeakers).compactMap { slot -> [String: Any]? in
                        let frames = (0..<prepared.segmentation.numFrames).filter {
                            prepared.segmentation.speakerWeights[chunk][$0][slot] > 0
                        }
                        return frames.isEmpty ? nil : ["chunk": chunk, "slot": slot, "activeFrames": frames]
                    }
                },
                "elapsedSeconds": Date().timeIntervalSince(start), "releaseAcceptance": false,
            ]
            let destination = output.appendingPathComponent(entry.fixtureID, isDirectory: true)
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
            try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]).write(
                to: destination.appendingPathComponent("report.json"), options: .withoutOverwriting
            )
        }
    }

    private struct Component: Decodable {
        let chunkIndex: Int
        let slot: Int
        let activeFrames: [Int]
        let embeddingRow: Int?
    }

    private struct Input: Decodable {
        let fixtureID: String
        let embeddings: [[Double]]
        let rho: [[Double]]
        let phi: [Double]
        let trainingRows: [Int]
        let expectedAHC: [Int]
        let expectedFinalLabels: [Int]
        let cleanIntervals: [[[Int]]]
        let minimumSharedSamples: Int
        let numChunks: Int
        let numFrames: Int
        let numSlots: Int
        let chunkOffsets: [Double]
        let frameDuration: Double
        let components: [Component]
    }

    private struct Manifest: Decodable {
        struct Case: Decodable {
            let fixtureID: String
            let input: String
        }
        let cases: [Case]
        let pldaModel: String
    }

    func testNativeClusteringAndReconstructionOfCompleteCorpus() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let inputPath = environment["PARASPEECH_SPEAKER_REPAIR_INPUT"],
            let outputPath = environment["PARASPEECH_SPEAKER_REPAIR_OUTPUT"]
        else {
            throw XCTSkip("Explicit frozen speaker repair corpus paths are required")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: inputPath)))
        XCTAssertEqual(manifest.cases.count, 12)
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: URL(fileURLWithPath: manifest.pldaModel), configuration: modelConfiguration)
        var configuration = OfflineDiarizerConfig(clusteringThreshold: 0.75)
        configuration.segmentationStepRatio = 0.1
        configuration.useSpeechComponentEmbeddings = true
        for entry in manifest.cases {
            try autoreleasepool {
                let input = try JSONDecoder().decode(
                    Input.self, from: Data(contentsOf: URL(fileURLWithPath: entry.input)))
                XCTAssertEqual(input.fixtureID, entry.fixtureID)
                let training = input.trainingRows.map { input.embeddings[$0] }
                let rho = input.trainingRows.map { input.rho[$0] }
                let ahc = AHCClustering().cluster(
                    embeddingFeatures: training, threshold: configuration.clusteringThreshold)
                XCTAssertEqual(ahc, input.expectedAHC, input.fixtureID)
                let vbx = VBxClustering(
                    config: configuration,
                    pldaTransform: PLDATransform(pldaRhoModel: model, psi: input.phi)
                ).refine(rhoFeatures: rho, initialClusters: ahc)
                let manager = OfflineDiarizerManager(config: configuration)
                let initial = manager.computeCentroids(
                    trainingEmbeddings: training, vbxOutput: vbx, initialClusters: ahc)
                let columns = initial.mapping.sorted { $0.value < $1.value }.map(\.key)
                let centroids = try OfflineSpeakerMergeSupport.refine(
                    centroids: initial.centroids, retainedColumns: columns, gamma: vbx.gamma,
                    initialClusters: ahc, trainingEmbeddings: training,
                    cleanIntervals: input.cleanIntervals.map { $0.map { $0[0]..<$0[1] } },
                    minimumSharedSamples: input.minimumSharedSamples
                )
                let labels = manager.assignEmbeddings(embeddingFeatures: input.embeddings, centroids: centroids)
                XCTAssertEqual(labels, input.expectedFinalLabels, input.fixtureID)
                var weights = Array(
                    repeating: Array(
                        repeating: Array(repeating: Float.zero, count: input.numSlots), count: input.numFrames),
                    count: input.numChunks)
                var assignments = Array(repeating: Array(repeating: -2, count: input.numSlots), count: input.numChunks)
                for component in input.components {
                    for frame in component.activeFrames { weights[component.chunkIndex][frame][component.slot] = 1 }
                    if let row = component.embeddingRow {
                        assignments[component.chunkIndex][component.slot] = labels[row]
                    }
                }
                let reconstructed = OfflineReconstruction(config: configuration).buildSegmentOutputs(
                    segmentation: SegmentationOutput(
                        logProbs: [], speakerWeights: weights, numChunks: input.numChunks,
                        numFrames: input.numFrames, numSpeakers: input.numSlots,
                        chunkOffsets: input.chunkOffsets, frameDuration: input.frameDuration
                    ),
                    hardClusters: assignments, centroids: centroids
                )
                func segments(_ values: [TimedSpeakerSegment]) -> [[String: Any]] {
                    values.map {
                        [
                            "speakerID": $0.speakerId, "startTime": $0.startTimeSeconds, "endTime": $0.endTimeSeconds,
                            "qualityScore": $0.qualityScore,
                        ]
                    }
                }
                let report: [String: Any] = [
                    "status": "completed", "kind": "shipping_rule_component_arithmetic_control",
                    "fixtureID": input.fixtureID, "speakerCount": centroids.count,
                    "segments": segments(reconstructed.segments),
                    "activitySegments": segments(reconstructed.speakerActivitySegments),
                    "ahcLabels": ahc, "finalLabels": labels,
                    "nativeSwiftClusteringExecuted": true, "appIntegrationExecuted": false,
                    "releaseAcceptance": false,
                ]
                let destination = output.appendingPathComponent(input.fixtureID, isDirectory: true)
                try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
                try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]).write(
                    to: destination.appendingPathComponent("report.json"), options: .withoutOverwriting
                )
            }
        }
    }
}
