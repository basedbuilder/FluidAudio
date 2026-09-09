import Foundation

/// Preserves distinct AHC seeds when a proposed VBx merge lacks acoustic support.
/// Inputs belong to one physical recording; clean intervals use its sample clock.
struct OfflineSpeakerMergeSupport {
    static func refine(
        centroids: [[Double]],
        retainedColumns: [Int],
        gamma: [[Double]],
        initialClusters: [Int],
        trainingEmbeddings: [[Double]],
        cleanIntervals: [[Range<Int>]],
        minimumSharedSamples: Int,
        maximumCosineDistance: Double = 0.65
    ) throws -> [[Double]] {
        let count = trainingEmbeddings.count
        guard count > 0, let dimension = trainingEmbeddings.first?.count,
            dimension > 0, initialClusters.count == count,
            gamma.count == count, cleanIntervals.count == count,
            centroids.count == retainedColumns.count, !centroids.isEmpty,
            minimumSharedSamples > 0, (0...2).contains(maximumCosineDistance),
            initialClusters.allSatisfy({ $0 >= 0 }),
            trainingEmbeddings.allSatisfy({ $0.count == dimension && $0.allSatisfy(\.isFinite) }),
            centroids.allSatisfy({ $0.count == dimension && $0.allSatisfy(\.isFinite) }),
            gamma.allSatisfy({ row in
                retainedColumns.allSatisfy { row.indices.contains($0) && row[$0].isFinite && row[$0] >= 0 }
            }),
            cleanIntervals.allSatisfy({ intervals in
                !intervals.isEmpty && intervals.allSatisfy({ !$0.isEmpty && $0.lowerBound >= 0 })
                    && zip(intervals, intervals.dropFirst()).allSatisfy({ $0.upperBound <= $1.lowerBound })
            })
        else {
            throw OfflineDiarizationError.invalidConfiguration("Invalid speaker merge evidence")
        }

        let seeds = Array(Set(initialClusters)).sorted()
        let indexBySeed = Dictionary(uniqueKeysWithValues: seeds.enumerated().map { ($1, $0) })
        let rowsBySeed = seeds.map { seed in initialClusters.indices.filter { initialClusters[$0] == seed } }
        var proposed = [Int]()
        var means = [[Double]]()
        for rows in rowsBySeed {
            var scores = [Double](repeating: 0, count: retainedColumns.count)
            var mean = [Double](repeating: 0, count: dimension)
            for row in rows {
                for column in retainedColumns.indices { scores[column] += gamma[row][retainedColumns[column]] }
                for axis in 0..<dimension { mean[axis] += trainingEmbeddings[row][axis] }
            }
            proposed.append(scores.indices.dropFirst().reduce(0) { scores[$1] > scores[$0] ? $1 : $0 })
            means.append(mean.map { $0 / Double(rows.count) })
        }

        var parents = Array(seeds.indices)
        func root(_ index: Int) -> Int {
            var current = index
            while parents[current] != current {
                parents[current] = parents[parents[current]]
                current = parents[current]
            }
            return current
        }
        func connect(_ left: Int, _ right: Int) {
            guard left != right, proposed[left] == proposed[right] else { return }
            let leftRoot = root(left)
            let rightRoot = root(right)
            parents[rightRoot] = leftRoot
        }

        for left in seeds.indices {
            for right in seeds.indices where right > left && proposed[left] == proposed[right] {
                let lhs = means[left]
                let rhs = means[right]
                let dot = zip(lhs, rhs).reduce(0) { $0 + $1.0 * $1.1 }
                let norm = sqrt(lhs.reduce(0) { $0 + $1 * $1 } * rhs.reduce(0) { $0 + $1 * $1 })
                if norm > 0, 1 - dot / norm < maximumCosineDistance { connect(left, right) }
            }
        }

        // Window copies overlap only nearby on the physical clock. Avoid comparing
        // every pair in a long meeting just to discover disjoint sample ranges.
        let orderedRows = cleanIntervals.indices.sorted {
            cleanIntervals[$0][0].lowerBound < cleanIntervals[$1][0].lowerBound
        }
        for (position, leftRow) in orderedRows.enumerated() {
            guard let leftSeed = indexBySeed[initialClusters[leftRow]],
                let end = cleanIntervals[leftRow].last?.upperBound
            else { continue }
            for rightRow in orderedRows.dropFirst(position + 1) {
                if cleanIntervals[rightRow][0].lowerBound > end - minimumSharedSamples { break }
                guard let rightSeed = indexBySeed[initialClusters[rightRow]],
                    leftSeed != rightSeed, proposed[leftSeed] == proposed[rightSeed]
                else { continue }
                if sharedSamples(cleanIntervals[leftRow], cleanIntervals[rightRow]) >= minimumSharedSamples {
                    connect(leftSeed, rightSeed)
                }
            }
        }

        var result = [[Double]]()
        for column in centroids.indices {
            var groups = [[Int]]()
            var groupIndexByRoot = [Int: Int]()
            for seed in seeds.indices where proposed[seed] == column {
                let owner = root(seed)
                if let index = groupIndexByRoot[owner] {
                    groups[index].append(seed)
                } else {
                    groupIndexByRoot[owner] = groups.count
                    groups.append([seed])
                }
            }
            guard groups.count > 1 else {
                result.append(centroids[column])
                continue
            }
            for group in groups {
                var numerator = [Double](repeating: 0, count: dimension)
                var denominator = 0.0
                for seed in group {
                    for row in rowsBySeed[seed] {
                        let weight = gamma[row][retainedColumns[column]]
                        denominator += weight
                        for axis in 0..<dimension { numerator[axis] += weight * trainingEmbeddings[row][axis] }
                    }
                }
                guard denominator > 0, denominator.isFinite else {
                    throw OfflineDiarizationError.processingFailed("Speaker merge evidence has no posterior support")
                }
                result.append(numerator.map { $0 / denominator })
            }
        }
        return result
    }

    private static func sharedSamples(_ left: [Range<Int>], _ right: [Range<Int>]) -> Int {
        var i = 0
        var j = 0
        var total = 0
        while i < left.count && j < right.count {
            total += max(0, min(left[i].upperBound, right[j].upperBound) - max(left[i].lowerBound, right[j].lowerBound))
            if left[i].upperBound < right[j].upperBound { i += 1 } else { j += 1 }
        }
        return total
    }
}
