import Foundation

struct ManifestDownloadRunner {
    private let downloader: ByteCountingFileDownloader

    init(downloader: ByteCountingFileDownloader = ByteCountingFileDownloader()) {
        self.downloader = downloader
    }

    func run(
        units: [ManifestDownloadUnit],
        legacyFractionMultiplier: Double,
        progress: @escaping @Sendable (DownloadUtils.DownloadProgress) -> Void
    ) async throws {
        let entries = units.flatMap { unit in
            unit.manifest.files.map { file in
                ManifestDownloadEntry(
                    unit: unit,
                    file: file,
                    destinationURL: unit.destinationRoot.appendingPathComponent(file.localRelativePath)
                )
            }
        }
        let totalFiles = entries.count
        let totalKnownBytes = units.reduce(0) { $0 + $1.manifest.totalKnownBytes }
        let hasUnknownSizes = units.contains { $0.manifest.hasUnknownSizes }
        var completedFiles = 0
        var completedBytes: Int64 = 0
        var completeDestinations = Set<String>()

        for entry in entries {
            if let existingBytes = try existingCompleteBytes(file: entry.file, destinationURL: entry.destinationURL) {
                completedFiles += 1
                completedBytes += existingBytes
                completeDestinations.insert(entry.destinationURL.path)
            }
        }

        progress(
            makeProgress(
                completedBytes: completedBytes,
                totalKnownBytes: totalKnownBytes,
                hasUnknownSizes: hasUnknownSizes,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                currentFile: nil,
                legacyFractionMultiplier: legacyFractionMultiplier
            ))

        for entry in entries {
            guard !completeDestinations.contains(entry.destinationURL.path) else { continue }

            try DownloadUtils.createDirectoryRobustly(at: entry.destinationURL.deletingLastPathComponent())

            if entry.file.sizeBytes == 0 {
                FileManager.default.createFile(atPath: entry.destinationURL.path, contents: Data())
                completedFiles += 1
                progress(
                    makeProgress(
                        completedBytes: completedBytes,
                        totalKnownBytes: totalKnownBytes,
                        hasUnknownSizes: hasUnknownSizes,
                        completedFiles: completedFiles,
                        totalFiles: totalFiles,
                        currentFile: entry.file.localRelativePath,
                        legacyFractionMultiplier: legacyFractionMultiplier
                    ))
                continue
            }

            let request = try entry.unit.makeRequest(entry.file)
            let baseCompletedBytes = completedBytes
            let completedFilesBeforeCurrent = completedFiles
            let result = try await downloader.download(
                request: request,
                to: entry.destinationURL,
                expectedBytes: entry.file.sizeBytes,
                progress: { fileBytes in
                    progress(
                        makeProgress(
                            completedBytes: baseCompletedBytes + fileBytes,
                            totalKnownBytes: totalKnownBytes,
                            hasUnknownSizes: hasUnknownSizes,
                            completedFiles: completedFilesBeforeCurrent,
                            totalFiles: totalFiles,
                            currentFile: entry.file.localRelativePath,
                            legacyFractionMultiplier: legacyFractionMultiplier
                        ))
                }
            )

            completedFiles += 1
            completedBytes = baseCompletedBytes + result.bytesWritten
            progress(
                makeProgress(
                    completedBytes: completedBytes,
                    totalKnownBytes: totalKnownBytes,
                    hasUnknownSizes: hasUnknownSizes,
                    completedFiles: completedFiles,
                    totalFiles: totalFiles,
                    currentFile: entry.file.localRelativePath,
                    legacyFractionMultiplier: legacyFractionMultiplier
                ))
        }
    }

    private func existingCompleteBytes(file: DownloadFile, destinationURL: URL) throws -> Int64? {
        guard FileManager.default.fileExists(atPath: destinationURL.path) else {
            return nil
        }

        let existingBytes = try ByteCountingFileDownloader.fileSize(at: destinationURL)
        guard let expectedBytes = file.sizeBytes else {
            return existingBytes
        }

        if existingBytes == expectedBytes {
            return existingBytes
        }

        try FileManager.default.removeItem(at: destinationURL)
        return nil
    }

    private func makeProgress(
        completedBytes: Int64,
        totalKnownBytes: Int64,
        hasUnknownSizes: Bool,
        completedFiles: Int,
        totalFiles: Int,
        currentFile: String?,
        legacyFractionMultiplier: Double
    ) -> DownloadUtils.DownloadProgress {
        let byteFraction: Double?
        let totalBytes: Int64?
        if hasUnknownSizes || totalKnownBytes <= 0 {
            byteFraction = nil
            totalBytes = nil
        } else {
            byteFraction = min(Double(completedBytes) / Double(totalKnownBytes), 1.0)
            totalBytes = totalKnownBytes
        }

        let fallbackFraction = totalFiles == 0 ? 1.0 : Double(completedFiles) / Double(totalFiles)
        let legacyFraction = legacyFractionMultiplier * (byteFraction ?? fallbackFraction)

        return DownloadUtils.DownloadProgress(
            fractionCompleted: min(legacyFraction, legacyFractionMultiplier),
            phase: .downloading(completedFiles: completedFiles, totalFiles: totalFiles),
            downloadedBytes: completedBytes,
            totalBytes: totalBytes,
            downloadFractionCompleted: byteFraction,
            currentFile: currentFile
        )
    }
}

private struct ManifestDownloadEntry {
    let unit: ManifestDownloadUnit
    let file: DownloadFile
    let destinationURL: URL
}
