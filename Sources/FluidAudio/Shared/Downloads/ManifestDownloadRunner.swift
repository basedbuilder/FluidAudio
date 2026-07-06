import Foundation
import os.lock

struct ManifestDownloadRunner {
    private static let logger = AppLogger(category: "ManifestDownloadRunner")
    private static let maxConcurrentDownloads = 6
    private static let maxDownloadAttempts = 4
    private static let minRetryBackoff: TimeInterval = 1.0

    typealias FileDownloadOperation =
        @Sendable (
            URLRequest,
            URL,
            Int64?,
            @escaping @Sendable (Int64) -> Void
        ) async throws -> ValidatedDownloadResult

    private let downloadFile: FileDownloadOperation
    private let retrySleep: @Sendable (TimeInterval) async throws -> Void

    init(downloader: ByteCountingFileDownloader = ByteCountingFileDownloader()) {
        self.downloadFile = { request, destinationURL, expectedBytes, progress in
            try await downloader.download(
                request: request,
                to: destinationURL,
                expectedBytes: expectedBytes,
                progress: progress
            )
        }
        self.retrySleep = Self.defaultRetrySleep
    }

    init(downloadFile: @escaping FileDownloadOperation) {
        self.downloadFile = downloadFile
        self.retrySleep = Self.defaultRetrySleep
    }

    init(
        downloadFile: @escaping FileDownloadOperation,
        retrySleep: @escaping @Sendable (TimeInterval) async throws -> Void
    ) {
        self.downloadFile = downloadFile
        self.retrySleep = retrySleep
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
        var pendingEntries: [ManifestDownloadEntry] = []
        let startedAt = Date()

        for entry in entries {
            if let existingBytes = try existingCompleteBytes(file: entry.file, destinationURL: entry.destinationURL) {
                completedFiles += 1
                completedBytes += existingBytes
                completeDestinations.insert(entry.destinationURL.path)
            }
        }

        let progressState = ManifestDownloadProgressState(
            completedFiles: completedFiles,
            completedBytes: completedBytes,
            totalKnownBytes: totalKnownBytes,
            hasUnknownSizes: hasUnknownSizes,
            totalFiles: totalFiles,
            legacyFractionMultiplier: legacyFractionMultiplier
        )
        progress(progressState.snapshot(currentFile: nil))

        for entry in entries {
            guard !completeDestinations.contains(entry.destinationURL.path) else { continue }

            try DownloadUtils.createDirectoryRobustly(at: entry.destinationURL.deletingLastPathComponent())

            if entry.file.sizeBytes == 0 {
                FileManager.default.createFile(atPath: entry.destinationURL.path, contents: Data())
                progress(
                    progressState.complete(
                        fileKey: entry.fileKey,
                        bytesWritten: 0,
                        currentFile: entry.file.localRelativePath
                    ))
                continue
            }

            pendingEntries.append(entry)
        }

        Self.logger.info(
            "Manifest download prepared total_files=\(totalFiles) pending_files=\(pendingEntries.count) skipped_files=\(completedFiles) total_known_bytes=\(totalKnownBytes) completed_bytes=\(completedBytes) unknown_sizes=\(hasUnknownSizes) max_concurrent=\(Self.maxConcurrentDownloads)"
        )

        try await downloadPendingEntries(
            pendingEntries,
            progressState: progressState,
            progress: progress
        )
        let elapsed = Self.formatElapsed(since: startedAt)
        let downloadedBytes = progressState.downloadedBytes
        Self.logger.info(
            "Manifest download finished total_files=\(totalFiles) downloaded_bytes=\(downloadedBytes) elapsed_s=\(elapsed) average_mbps=\(Self.formatMbps(bytes: downloadedBytes - completedBytes, since: startedAt))"
        )
    }

    private func downloadPendingEntries(
        _ entries: [ManifestDownloadEntry],
        progressState: ManifestDownloadProgressState,
        progress: @escaping @Sendable (DownloadUtils.DownloadProgress) -> Void
    ) async throws {
        guard !entries.isEmpty else { return }

        try await withThrowingTaskGroup(of: ManifestDownloadCompletion.self) { group in
            var nextIndex = 0

            func scheduleNext() {
                guard nextIndex < entries.count else { return }
                let entry = entries[nextIndex]
                nextIndex += 1

                group.addTask {
                    let request = try entry.unit.makeRequest(entry.file)
                    let startedAt = Date()
                    Self.logger.info(
                        "Manifest file download starting path=\(entry.file.localRelativePath) bytes=\(entry.file.sizeBytes ?? -1)"
                    )
                    let result = try await downloadWithTransientRetry(
                        request,
                        destinationURL: entry.destinationURL,
                        expectedBytes: entry.file.sizeBytes
                    ) { fileBytes in
                        progress(
                            progressState.updateInProgress(
                                fileKey: entry.fileKey,
                                bytesWritten: fileBytes,
                                currentFile: entry.file.localRelativePath
                            ))
                    }
                    Self.logger.info(
                        "Manifest file download finished path=\(entry.file.localRelativePath) bytes=\(result.bytesWritten) elapsed_s=\(Self.formatElapsed(since: startedAt)) average_mbps=\(Self.formatMbps(bytes: result.bytesWritten, since: startedAt))"
                    )
                    return ManifestDownloadCompletion(
                        fileKey: entry.fileKey,
                        currentFile: entry.file.localRelativePath,
                        bytesWritten: result.bytesWritten
                    )
                }
            }

            for _ in 0..<min(Self.maxConcurrentDownloads, entries.count) {
                scheduleNext()
            }

            while let completion = try await group.next() {
                progress(
                    progressState.complete(
                        fileKey: completion.fileKey,
                        bytesWritten: completion.bytesWritten,
                        currentFile: completion.currentFile
                    ))
                scheduleNext()
            }
        }
    }

    private func downloadWithTransientRetry(
        _ request: URLRequest,
        destinationURL: URL,
        expectedBytes: Int64?,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws -> ValidatedDownloadResult {
        var lastError: Error?

        for attempt in 1...Self.maxDownloadAttempts {
            do {
                return try await downloadFile(request, destinationURL, expectedBytes, progress)
            } catch {
                lastError = error
                guard attempt < Self.maxDownloadAttempts, Self.isRetryableDownloadError(error) else {
                    throw error
                }

                let backoffSeconds = pow(2.0, Double(attempt - 1)) * Self.minRetryBackoff
                Self.logger.warning(
                    "Retrying file download path=\(destinationURL.lastPathComponent) attempt=\(attempt + 1) backoff_s=\(Self.formatSeconds(backoffSeconds)) error=\(error.localizedDescription)"
                )
                try await retrySleep(backoffSeconds)
            }
        }

        throw lastError ?? DownloadUtils.HuggingFaceDownloadError.invalidResponse
    }

    private static func defaultRetrySleep(_ seconds: TimeInterval) async throws {
        try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
    }

    private static func formatElapsed(since startedAt: Date) -> String {
        formatSeconds(max(Date().timeIntervalSince(startedAt), 0.001))
    }

    private static func formatSeconds(_ seconds: TimeInterval) -> String {
        String(format: "%.2f", seconds)
    }

    private static func formatMbps(bytes: Int64, since startedAt: Date) -> String {
        let elapsed = max(Date().timeIntervalSince(startedAt), 0.001)
        let mbps = Double(max(bytes, 0)) * 8 / elapsed / 1_000_000
        return String(format: "%.1f", mbps)
    }

    private static func isRetryableDownloadError(_ error: Error) -> Bool {
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut, .cannotConnectToHost, .cannotFindHost,
                .networkConnectionLost, .notConnectedToInternet,
                .dnsLookupFailed, .secureConnectionFailed,
                .resourceUnavailable:
                return true
            default:
                return false
            }
        }

        switch error {
        case DownloadUtils.HuggingFaceDownloadError.rateLimited:
            return true
        case DownloadUtils.HuggingFaceDownloadError.downloadFailed(_, let underlying):
            let nsError = underlying as NSError
            return (nsError.domain == "HTTP" && (500...599).contains(nsError.code))
                || Self.isRetryableByteCountingError(nsError)
        default:
            return false
        }
    }

    private static func isRetryableByteCountingError(_ error: NSError) -> Bool {
        guard error.domain == "FluidAudio.ByteCountingFileDownloader" else { return false }
        return [1, 3, 4].contains(error.code)
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

}

private struct ManifestDownloadEntry: Sendable {
    let unit: ManifestDownloadUnit
    let file: DownloadFile
    let destinationURL: URL

    var fileKey: String { destinationURL.path }
}

private struct ManifestDownloadCompletion: Sendable {
    let fileKey: String
    let currentFile: String
    let bytesWritten: Int64
}

private final class ManifestDownloadProgressState: Sendable {
    private struct State {
        var completedFiles: Int
        var completedBytes: Int64
        var inProgressBytes: [String: Int64] = [:]
    }

    private let totalKnownBytes: Int64
    private let hasUnknownSizes: Bool
    private let totalFiles: Int
    private let legacyFractionMultiplier: Double
    private let state: OSAllocatedUnfairLock<State>

    init(
        completedFiles: Int,
        completedBytes: Int64,
        totalKnownBytes: Int64,
        hasUnknownSizes: Bool,
        totalFiles: Int,
        legacyFractionMultiplier: Double
    ) {
        self.totalKnownBytes = totalKnownBytes
        self.hasUnknownSizes = hasUnknownSizes
        self.totalFiles = totalFiles
        self.legacyFractionMultiplier = legacyFractionMultiplier
        self.state = OSAllocatedUnfairLock(
            initialState: State(
                completedFiles: completedFiles,
                completedBytes: completedBytes
            ))
    }

    func updateInProgress(
        fileKey: String,
        bytesWritten: Int64,
        currentFile: String?
    ) -> DownloadUtils.DownloadProgress {
        state.withLock { state in
            state.inProgressBytes[fileKey] = max(bytesWritten, 0)
            return makeProgress(state: state, currentFile: currentFile)
        }
    }

    func snapshot(currentFile: String?) -> DownloadUtils.DownloadProgress {
        state.withLock { state in
            makeProgress(state: state, currentFile: currentFile)
        }
    }

    func complete(
        fileKey: String,
        bytesWritten: Int64,
        currentFile: String?
    ) -> DownloadUtils.DownloadProgress {
        state.withLock { state in
            state.inProgressBytes[fileKey] = nil
            state.completedFiles += 1
            state.completedBytes += max(bytesWritten, 0)
            return makeProgress(state: state, currentFile: currentFile)
        }
    }

    var downloadedBytes: Int64 {
        state.withLock { state in
            state.completedBytes + state.inProgressBytes.values.reduce(0, +)
        }
    }

    private func makeProgress(
        state: State,
        currentFile: String?
    ) -> DownloadUtils.DownloadProgress {
        let activeBytes = state.inProgressBytes.values.reduce(0, +)
        let downloadedBytes = state.completedBytes + activeBytes
        let byteFraction: Double?
        let totalBytes: Int64?
        if hasUnknownSizes || totalKnownBytes <= 0 {
            byteFraction = nil
            totalBytes = nil
        } else {
            byteFraction = min(Double(downloadedBytes) / Double(totalKnownBytes), 1.0)
            totalBytes = totalKnownBytes
        }

        let fallbackFraction = totalFiles == 0 ? 1.0 : Double(state.completedFiles) / Double(totalFiles)
        let legacyFraction = legacyFractionMultiplier * (byteFraction ?? fallbackFraction)

        return DownloadUtils.DownloadProgress(
            fractionCompleted: min(legacyFraction, legacyFractionMultiplier),
            phase: .downloading(completedFiles: state.completedFiles, totalFiles: totalFiles),
            downloadedBytes: downloadedBytes,
            totalBytes: totalBytes,
            downloadFractionCompleted: byteFraction,
            currentFile: currentFile
        )
    }
}
