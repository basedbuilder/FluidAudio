import Foundation

/// Consolidates AHC identities that are duplicate observations of the same clean
/// speech in overlapping segmentation windows.
///
/// This helper is intentionally source-local. It receives one diarizer manager's
/// embeddings and never compares persisted identities, lexical content, or tracks.
struct OfflineTemporalClusterConsolidator {
    private static let activityThreshold: Float = 1e-3

    static func consolidate(
        timedEmbeddings: [TimedEmbedding],
        initialClusters: [Int],
        segmentation: SegmentationOutput,
        minimumSharedCleanDurationSeconds: Double = 0
    ) -> [Int] {
        guard
            !timedEmbeddings.isEmpty,
            timedEmbeddings.count == initialClusters.count,
            initialClusters.allSatisfy({ $0 >= 0 }),
            segmentation.frameDuration.isFinite,
            segmentation.frameDuration > 0,
            segmentation.numChunks > 1,
            segmentation.numSpeakers > 0
        else {
            return initialClusters
        }

        let evidence = mustLinkEvidence(
            timedEmbeddings: timedEmbeddings,
            clusters: initialClusters,
            segmentation: segmentation,
            minimumSharedCleanDurationSeconds:
                minimumSharedCleanDurationSeconds
        )
        guard !evidence.isEmpty else { return initialClusters }

        let conflicts = cannotLinkPairs(
            timedEmbeddings: timedEmbeddings,
            clusters: initialClusters,
            segmentation: segmentation
        )
        let clusterCount = (initialClusters.max() ?? -1) + 1
        guard clusterCount > 0 else { return initialClusters }

        var disjointSet = ConflictAwareDisjointSet(
            clusterCount: clusterCount,
            conflicts: conflicts
        )
        let orderedEdges =
            evidence
            .filter { $0.value.score.isFinite && $0.value.score > 0 }
            .sorted { lhs, rhs in
                if lhs.value.score != rhs.value.score {
                    return lhs.value.score > rhs.value.score
                }
                if lhs.key.lower != rhs.key.lower {
                    return lhs.key.lower < rhs.key.lower
                }
                return lhs.key.upper < rhs.key.upper
            }

        var mergedAny = false
        for edge in orderedEdges {
            mergedAny =
                disjointSet.unionIfAllowed(edge.key.lower, edge.key.upper)
                || mergedAny
        }
        guard mergedAny else { return initialClusters }

        var remappedRoots: [Int: Int] = [:]
        var nextLabel = 0
        return initialClusters.map { cluster in
            let root = disjointSet.find(cluster)
            if let existing = remappedRoots[root] {
                return existing
            }
            let label = nextLabel
            nextLabel += 1
            remappedRoots[root] = label
            return label
        }
    }

    private static func mustLinkEvidence(
        timedEmbeddings: [TimedEmbedding],
        clusters: [Int],
        segmentation: SegmentationOutput,
        minimumSharedCleanDurationSeconds: Double
    ) -> [ClusterPair: MustLinkEvidence] {
        let requiredFrameCount = minimumSharedFrameCount(
            durationSeconds: minimumSharedCleanDurationSeconds,
            frameDuration: segmentation.frameDuration
        )
        let positionsByChunk = Dictionary(
            grouping: timedEmbeddings.indices,
            by: { timedEmbeddings[$0].chunkIndex }
        )
        let orderedChunks = positionsByChunk.keys.sorted()
        var priorObservations: [Int64: [FrameObservation]] = [:]
        var evidence: [ClusterPair: MustLinkEvidence] = [:]

        for chunkIndex in orderedChunks {
            guard
                chunkIndex >= 0,
                chunkIndex < segmentation.numChunks,
                segmentation.chunkOffsets.indices.contains(chunkIndex),
                segmentation.chunkOffsets[chunkIndex].isFinite,
                let chunkPositions = positionsByChunk[chunkIndex]
            else {
                continue
            }

            let chunkStart = globalFrame(
                chunkOffset: segmentation.chunkOffsets[chunkIndex],
                localFrame: 0,
                frameDuration: segmentation.frameDuration
            )
            if let chunkStart {
                priorObservations = priorObservations.filter {
                    $0.key >= chunkStart
                }
            }

            var currentObservations: [Int64: [FrameObservation]] = [:]
            for position in chunkPositions.sorted(by: {
                let lhs = timedEmbeddings[$0]
                let rhs = timedEmbeddings[$1]
                if lhs.startFrame != rhs.startFrame {
                    return lhs.startFrame < rhs.startFrame
                }
                if lhs.speakerIndex != rhs.speakerIndex {
                    return lhs.speakerIndex < rhs.speakerIndex
                }
                return $0 < $1
            }) {
                let embedding = timedEmbeddings[position]
                let cluster = clusters[position]
                for localFrame in embedding.frameWeights.indices {
                    let weight = embedding.frameWeights[localFrame]
                    guard
                        weight.isFinite,
                        weight > activityThreshold,
                        let frame = globalFrame(
                            chunkOffset: segmentation.chunkOffsets[chunkIndex],
                            localFrame: localFrame,
                            frameDuration: segmentation.frameDuration
                        )
                    else {
                        continue
                    }
                    currentObservations[frame, default: []].append(
                        FrameObservation(cluster: cluster, weight: weight)
                    )
                }
            }

            for frame in currentObservations.keys.sorted() {
                guard
                    let current = currentObservations[frame],
                    let prior = priorObservations[frame]
                else {
                    continue
                }
                for currentObservation in current {
                    for priorObservation in prior
                    where currentObservation.cluster != priorObservation.cluster {
                        let pair = ClusterPair(
                            currentObservation.cluster,
                            priorObservation.cluster
                        )
                        evidence[pair, default: MustLinkEvidence()].add(
                            frame: frame,
                            weight: min(
                                currentObservation.weight,
                                priorObservation.weight
                            ),
                            requiredFrameCount: requiredFrameCount
                        )
                    }
                }
            }

            for frame in currentObservations.keys.sorted() {
                priorObservations[frame, default: []].append(
                    contentsOf: currentObservations[frame] ?? []
                )
            }
        }

        return evidence.filter { $0.value.isQualified }
    }

    private static func minimumSharedFrameCount(
        durationSeconds: Double,
        frameDuration: Double
    ) -> Int {
        guard
            durationSeconds.isFinite,
            durationSeconds > 0,
            frameDuration.isFinite,
            frameDuration > 0
        else {
            return 2
        }
        let rawFrameCount = (durationSeconds / frameDuration).rounded(.up)
        guard rawFrameCount.isFinite, rawFrameCount < Double(Int.max) else {
            return Int.max
        }
        return max(2, Int(rawFrameCount))
    }

    private static func cannotLinkPairs(
        timedEmbeddings: [TimedEmbedding],
        clusters: [Int],
        segmentation: SegmentationOutput
    ) -> Set<ClusterPair> {
        var runs = Array(
            repeating: Array(
                repeating: [RunCluster](),
                count: segmentation.numSpeakers
            ),
            count: segmentation.numChunks
        )

        for (embedding, cluster) in zip(timedEmbeddings, clusters) {
            guard
                runs.indices.contains(embedding.chunkIndex),
                runs[embedding.chunkIndex].indices.contains(embedding.speakerIndex),
                embedding.startFrame >= 0,
                embedding.endFrame >= embedding.startFrame
            else {
                continue
            }
            runs[embedding.chunkIndex][embedding.speakerIndex].append(
                RunCluster(
                    frames: embedding.startFrame...embedding.endFrame,
                    cluster: cluster
                )
            )
        }

        var conflicts: Set<ClusterPair> = []
        for chunkIndex in 0..<min(segmentation.numChunks, segmentation.speakerWeights.count) {
            let chunkWeights = segmentation.speakerWeights[chunkIndex]
            for (frameIndex, frameWeights) in chunkWeights.enumerated() {
                let activeSpeakers = frameWeights.indices.filter {
                    $0 < segmentation.numSpeakers
                        && frameWeights[$0].isFinite
                        && frameWeights[$0] > activityThreshold
                }
                guard activeSpeakers.count > 1 else { continue }

                for firstOffset in 0..<(activeSpeakers.count - 1) {
                    for secondOffset in (firstOffset + 1)..<activeSpeakers.count {
                        let firstSpeaker = activeSpeakers[firstOffset]
                        let secondSpeaker = activeSpeakers[secondOffset]
                        let firstClusters = runs[chunkIndex][firstSpeaker].compactMap {
                            $0.frames.contains(frameIndex) ? $0.cluster : nil
                        }
                        let secondClusters = runs[chunkIndex][secondSpeaker].compactMap {
                            $0.frames.contains(frameIndex) ? $0.cluster : nil
                        }
                        for firstCluster in firstClusters {
                            for secondCluster in secondClusters
                            where firstCluster != secondCluster {
                                conflicts.insert(
                                    ClusterPair(firstCluster, secondCluster)
                                )
                            }
                        }
                    }
                }
            }
        }
        return conflicts
    }

    private static func globalFrame(
        chunkOffset: Double,
        localFrame: Int,
        frameDuration: Double
    ) -> Int64? {
        let value = chunkOffset / frameDuration + Double(localFrame)
        guard
            value.isFinite,
            value >= Double(Int64.min),
            value <= Double(Int64.max)
        else {
            return nil
        }
        return Int64(value.rounded())
    }
}

private struct FrameObservation {
    let cluster: Int
    let weight: Float
}

private struct MustLinkEvidence {
    private(set) var score = 0.0
    private(set) var isQualified = false
    private var uniqueFrames: Set<Int64> = []

    mutating func add(
        frame: Int64,
        weight: Float,
        requiredFrameCount: Int
    ) {
        score += Double(weight)
        guard !isQualified else { return }
        uniqueFrames.insert(frame)
        if uniqueFrames.count >= requiredFrameCount {
            isQualified = true
            uniqueFrames.removeAll(keepingCapacity: false)
        }
    }
}

private struct RunCluster {
    let frames: ClosedRange<Int>
    let cluster: Int
}

private struct ClusterPair: Hashable {
    let lower: Int
    let upper: Int

    init(_ first: Int, _ second: Int) {
        lower = min(first, second)
        upper = max(first, second)
    }
}

private struct ConflictAwareDisjointSet {
    private var parent: [Int]
    private var conflicts: [Set<Int>]

    init(clusterCount: Int, conflicts initialConflicts: Set<ClusterPair>) {
        parent = Array(0..<clusterCount)
        conflicts = Array(repeating: [], count: clusterCount)
        for pair in initialConflicts
        where parent.indices.contains(pair.lower)
            && parent.indices.contains(pair.upper)
            && pair.lower != pair.upper
        {
            conflicts[pair.lower].insert(pair.upper)
            conflicts[pair.upper].insert(pair.lower)
        }
    }

    mutating func find(_ value: Int) -> Int {
        guard parent.indices.contains(value) else { return value }
        var root = value
        while parent[root] != root {
            root = parent[root]
        }
        var current = value
        while parent[current] != current {
            let next = parent[current]
            parent[current] = root
            current = next
        }
        return root
    }

    mutating func unionIfAllowed(_ first: Int, _ second: Int) -> Bool {
        let firstRoot = find(first)
        let secondRoot = find(second)
        guard
            firstRoot != secondRoot,
            parent.indices.contains(firstRoot),
            parent.indices.contains(secondRoot),
            !componentsConflict(firstRoot, secondRoot)
        else {
            return false
        }

        let retainedRoot = min(firstRoot, secondRoot)
        let removedRoot = max(firstRoot, secondRoot)
        let neighbors = Array(
            conflicts[retainedRoot]
                .union(conflicts[removedRoot])
                .subtracting([retainedRoot, removedRoot])
        )
        parent[removedRoot] = retainedRoot
        conflicts[retainedRoot].removeAll(keepingCapacity: true)
        conflicts[removedRoot].removeAll(keepingCapacity: false)

        for neighbor in neighbors {
            let neighborRoot = find(neighbor)
            guard neighborRoot != retainedRoot else { continue }
            conflicts[retainedRoot].insert(neighborRoot)
            conflicts[neighborRoot].remove(firstRoot)
            conflicts[neighborRoot].remove(secondRoot)
            conflicts[neighborRoot].insert(retainedRoot)
        }
        return true
    }

    private mutating func componentsConflict(
        _ firstRoot: Int,
        _ secondRoot: Int
    ) -> Bool {
        let firstNeighbors = Array(conflicts[firstRoot])
        if firstNeighbors.contains(where: { find($0) == secondRoot }) {
            return true
        }
        let secondNeighbors = Array(conflicts[secondRoot])
        return secondNeighbors.contains(where: { find($0) == firstRoot })
    }
}
