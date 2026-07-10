import Foundation
import os

/// The one per-file download unit for the download stack (#765 Wave 4):
/// bounded retry (RetryPolicy) around the streaming byte-range-resume
/// transport (#757), artifact validation, and atomic placement into the
/// cache. Used by both `ModelHub.download` and `download(subdirectory:)`.
enum FileDownloader {

    /// Historical log category retained deliberately across the 0.16 rename so
    /// existing `category == "DownloadUtils"` predicates keep capturing the
    /// whole download trail; renaming it is a separate, opt-in decision.
    private static let logger = AppLogger(category: "DownloadUtils")

    struct ParallelRangeConfiguration: Sendable {
        let threshold: Int
        let chunkSize: Int
        let maxConcurrent: Int

        static let `default` = ParallelRangeConfiguration(
            threshold: 64 * 1024 * 1024,
            chunkSize: 16 * 1024 * 1024,
            maxConcurrent: 8
        )
    }

    private struct ParallelRange: Sendable {
        let index: Int
        let start: Int
        let end: Int

        var byteCount: Int { end - start + 1 }
    }

    private enum ParallelRangeError: LocalizedError {
        case unsupported(String)

        var errorDescription: String? {
            switch self {
            case .unsupported(let reason):
                return "Parallel byte ranges are unavailable: \(reason)"
            }
        }
    }

    private struct ParallelRangeTaskOutcome: Sendable {
        let index: Int
        let error: (any Error)?
    }

    private final class ParallelRangeState: Sendable {
        private struct State {
            var bytesByRange: [Int64]
            var validator: String?
        }

        private let expectedBytes: Int64
        private let progress: (@Sendable (Int64, Int64) -> Void)?
        private let state: OSAllocatedUnfairLock<State>

        init(rangeCount: Int, expectedBytes: Int64, progress: (@Sendable (Int64, Int64) -> Void)?) {
            self.expectedBytes = expectedBytes
            self.progress = progress
            self.state = OSAllocatedUnfairLock(
                initialState: State(bytesByRange: Array(repeating: 0, count: rangeCount)))
        }

        func record(range: ParallelRange, bytesWritten: Int64) {
            state.withLock { state in
                state.bytesByRange[range.index] = max(
                    state.bytesByRange[range.index],
                    min(max(bytesWritten, 0), Int64(range.byteCount))
                )
                progress?(state.bytesByRange.reduce(0, +), expectedBytes)
            }
        }

        func accept(validator: String, path: String) throws {
            try state.withLock { state in
                if let existing = state.validator, existing != validator {
                    throw DownloadError.invalidArtifact(
                        path: path, reason: "parallel range validators did not match")
                }
                state.validator = validator
            }
        }

        func acceptedValidator() -> String? {
            state.withLock { $0.validator }
        }
    }

    /// What `ensure(file:from:at:)` did for a file.
    enum Outcome {
        /// Destination already existed; nothing fetched.
        case alreadyPresent
        /// Zero-byte file created locally (HF serves 500s for empty files).
        case createdEmpty
        /// Fetched from the network, validated, and moved into place.
        case downloaded
    }

    /// Ensure `file` from `repoRemotePath` exists at `destination`: skip when
    /// present, create empties locally, otherwise download with retry +
    /// byte-range resume (via `<destination>.partial`), validate, and move
    /// atomically into place.
    ///
    /// - Parameter recoveringBlockedPaths: When true, a regular file blocking
    ///   a parent-directory component is DELETED and replaced (the repo-cache
    ///   corrupt-recovery behavior). When false, a blocked path fails loudly
    ///   with the thrown filesystem error — callers outside the managed model
    ///   cache must not silently destroy user files.
    @discardableResult
    static func ensure(
        file: RemoteFile,
        from repoRemotePath: String,
        at destination: URL,
        recoveringBlockedPaths: Bool,
        configuration: URLSessionConfiguration? = nil,
        onBytes: (@Sendable (Int64, Int64) -> Void)? = nil
    ) async throws -> Outcome {
        if FileManager.default.fileExists(atPath: destination.path) {
            return .alreadyPresent
        }

        let parentDir = destination.deletingLastPathComponent()
        if recoveringBlockedPaths {
            // Create parent directory, removing any conflicting files in the path.
            try ModelCache.createDirectoryRobustly(at: parentDir)
        } else {
            try FileManager.default.createDirectory(
                at: parentDir, withIntermediateDirectories: true)
        }

        // HuggingFace returns 500 for 0-byte files — create empty file locally.
        if file.size == 0 {
            FileManager.default.createFile(atPath: destination.path, contents: Data())
            return .createdEmpty
        }

        let encodedPath =
            file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
        let fileURL = try ModelRegistry.resolveModel(repoRemotePath, encodedPath)
        let request = HFClient.authorizedRequest(url: fileURL)

        let tempURL = try await download(
            request: request,
            path: file.path,
            expectedSize: file.size,
            partialFileURL: destination.appendingPathExtension("partial"),
            onProgress: onBytes,
            configuration: configuration
        )

        if FileManager.default.fileExists(atPath: destination.path) {
            try? FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
        return .downloaded
    }

    /// `response` is nil when re-validating a fully-downloaded partial file from
    /// a previous run (no live response to check Content-Type against).
    static func validateDownloadedArtifact(
        at tempURL: URL,
        response: HTTPURLResponse?,
        path: String,
        expectedSize: Int
    ) throws {
        if let contentType = response?.value(forHTTPHeaderField: "Content-Type")?.lowercased(),
            contentType.contains("text/html")
        {
            throw DownloadError.invalidArtifact(
                path: path, reason: "server returned Content-Type: \(contentType)")
        }

        let actualSize =
            ((try? FileManager.default.attributesOfItem(atPath: tempURL.path))?[.size] as? Int) ?? 0
        if actualSize == 0 {
            throw DownloadError.invalidArtifact(path: path, reason: "empty file")
        }

        if let handle = try? FileHandle(forReadingFrom: tempURL) {
            defer { try? handle.close() }
            if HFClient.looksLikeHTML(handle.readData(ofLength: 512)) {
                throw DownloadError.invalidArtifact(
                    path: path, reason: "response body begins with HTML markup")
            }
        }

        // HuggingFace reports the exact (LFS-resolved) object size; a short body is truncation.
        if expectedSize > 0 && actualSize != expectedSize {
            throw DownloadError.invalidArtifact(
                path: path,
                reason: "size mismatch (expected \(expectedSize) bytes, got \(actualSize))")
        }
    }

    // MARK: - Small-file fetch with converged policy (#765 Wave 5)

    /// Fetch a small file into memory with the converged policy (#765 Wave 5):
    /// classified retry (permanent 4xx fails fast, 5xx/rate-limits retry with
    /// Retry-After pacing) and artifact validation (HTML error pages and empty
    /// bodies are rejected, never returned as content). Replaces
    /// `fetchFile`'s historical retry-everything loop.
    static func fetchData(
        from url: URL,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0,
        configuration: URLSessionConfiguration? = nil
    ) async throws -> Data {
        let request = HFClient.authorizedRequest(url: url)
        let session = configuration.map { URLSession(configuration: $0) } ?? HFClient.session
        defer {
            if configuration != nil { session.finishTasksAndInvalidate() }
        }

        return try await RetryPolicy.withRetry(
            label: description, maxAttempts: maxAttempts, minBackoff: minBackoff, logger: logger
        ) { _ in
            // Re-check per attempt, matching download(): flipping
            // offlineMode mid-operation stops at the next request.
            guard !HFClient.offlineMode else {
                throw DownloadError.networkDisabled(
                    operation: "fetchFile(\(description))")
            }

            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw DownloadError.invalidResponse
            }

            try HFClient.checkRateLimitForRetry(httpResponse, context: "fetching \(description)")

            guard (200..<300).contains(httpResponse.statusCode) else {
                throw DownloadError.downloadFailed(
                    path: description,
                    underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                )
            }

            // Reject bodies that are error pages or truncated-to-nothing —
            // a 200 carrying HTML must never be cached as content (#748).
            // Content-Type check mirrors validateDownloadedArtifact's, so a
            // text/html page without a sniffable prefix is caught here too.
            if let contentType = httpResponse.value(forHTTPHeaderField: "Content-Type")?.lowercased(),
                contentType.contains("text/html")
            {
                throw DownloadError.invalidArtifact(
                    path: description, reason: "server returned Content-Type: \(contentType)")
            }
            if data.isEmpty {
                throw DownloadError.invalidArtifact(
                    path: description, reason: "empty file")
            }
            if HFClient.looksLikeHTML(data) {
                throw DownloadError.invalidArtifact(
                    path: description, reason: "response body begins with HTML markup")
            }

            return data
        }
    }

    // MARK: - Streaming download to a persistent file

    /// Stream a download directly into `destination` so received bytes survive
    /// a mid-transfer drop (#757). Pure transport: the caller sets resume
    /// headers and validates the HTTP status. Body bytes are written only for
    /// 2xx — appended after `resumeOffset` on 206, from byte 0 otherwise.
    ///
    /// - Parameters:
    ///   - resumeOffset: Byte count a 206 appends after; also the base for
    ///     progress callbacks so resumed progress never dips.
    ///   - configuration: Session configuration override for tests.
    ///   - onProgress: `(totalBytesWritten, totalBytesExpected)`, both
    ///     including `resumeOffset`. Delegate-driven byte progress (#756).
    ///   - onResponse: Fires before any body byte is written — the resume path
    ///     persists the validator here so it survives a drop mid-body.
    /// Coverage flows through the `FileDownloader.download`
    /// forward (DownloadResumeTests); no test drives this directly.
    static func streamDownload(
        request: URLRequest,
        to destination: URL,
        resumeOffset: Int64 = 0,
        configuration: URLSessionConfiguration? = nil,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil,
        onResponse: (@Sendable (HTTPURLResponse) -> Void)? = nil,
        validateResponse: (@Sendable (HTTPURLResponse) throws -> Void)? = nil
    ) async throws -> HTTPURLResponse {
        let delegate = StreamingDownloadDelegate(
            destination: destination,
            resumeOffset: resumeOffset,
            onProgress: onProgress,
            onResponse: onResponse,
            validateResponse: validateResponse
        )
        // Dedicated session with delegate — one per download to avoid cross-talk.
        // One session per download to avoid delegate cross-talk; the
        // configuration comes from the shared HFClient session.
        let session = URLSession(
            configuration: configuration ?? HFClient.session.configuration,
            delegate: delegate,
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let task = session.dataTask(with: request)
                delegate.attach(continuation: continuation, task: task)
                task.resume()
            }
        } onCancel: {
            delegate.cancel()
        }
    }

    // MARK: - Per-file download with bounded retry and byte-range resume

    /// Sidecar storing the HTTP validator (ETag/Last-Modified) a partial file
    /// came from; written at response time so it survives a drop mid-body.
    static func resumeValidatorURL(for partialFileURL: URL) -> URL {
        partialFileURL.appendingPathExtension("etag")
    }

    private static func fileSize(at url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64) ?? 0
    }

    /// Remove a partial download and its validator sidecar.
    private static func clearPartialDownload(_ partialFileURL: URL) {
        try? FileManager.default.removeItem(at: partialFileURL)
        try? FileManager.default.removeItem(at: resumeValidatorURL(for: partialFileURL))
    }

    /// Resume validator for `If-Range`: a strong ETag, else Last-Modified.
    /// Weak ETags are unusable for `If-Range` (RFC 9110 §13.1.5); nil means
    /// resume is unsafe and the file restarts from 0.
    private static func resumeValidator(from response: HTTPURLResponse) -> String? {
        if let etag = response.value(forHTTPHeaderField: "ETag"), !etag.hasPrefix("W/") {
            return etag
        }
        return response.value(forHTTPHeaderField: "Last-Modified")
    }

    /// Download a single repo file with bounded exponential-backoff retry on
    /// transient failures (timeout/TLS/connectivity, HTTP 429/503/5xx; 4xx and
    /// non-network errors fail fast — see `RetryPolicy.isRetryable`).
    ///
    /// Resume (#757): bytes stream into `partialFileURL`, so a retry — or a new
    /// process — continues with `Range`/`If-Range` instead of restarting from
    /// byte 0. A 200 (range ignored / remote changed) restarts the file rather
    /// than splicing mixed-version bytes.
    ///
    /// - Parameters:
    ///   - partialFileURL: Where in-flight bytes live (e.g. `<dest>.partial`).
    ///     Pass nil for a unique temp location (in-run resume only).
    ///   - configuration: Session configuration override for tests.
    /// - Returns: The URL of a validated (2xx) fully-written download; the
    ///   caller moves it into the cache.
    /// Pinned via the `FileDownloader.download` forward
    /// (DownloadResumeTests drives resume/splice/416 behavior through it).
    static func download(
        request: URLRequest,
        path: String,
        expectedSize: Int,
        partialFileURL: URL? = nil,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0,
        configuration: URLSessionConfiguration? = nil,
        parallelRanges: ParallelRangeConfiguration = .default
    ) async throws -> URL {
        precondition(parallelRanges.threshold > 0)
        precondition(parallelRanges.chunkSize > 0)
        precondition(parallelRanges.maxConcurrent > 0)

        let defaultPartialURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("partial")
        let partialURL = partialFileURL ?? defaultPartialURL
        let validatorURL = resumeValidatorURL(for: partialURL)

        if expectedSize >= parallelRanges.threshold, fileSize(at: partialURL) == 0 {
            do {
                return try await downloadInParallelRanges(
                    request: request,
                    path: path,
                    expectedSize: expectedSize,
                    partialURL: partialURL,
                    validatorURL: validatorURL,
                    onProgress: onProgress,
                    maxAttempts: maxAttempts,
                    minBackoff: minBackoff,
                    configuration: configuration,
                    rangeConfiguration: parallelRanges
                )
            } catch ParallelRangeError.unsupported(let reason) {
                logger.warning(
                    "Parallel ranges unavailable for \(path); using resumable single stream: \(reason)")
            }
        }

        return try await RetryPolicy.withRetry(
            label: path, maxAttempts: maxAttempts, minBackoff: minBackoff, logger: logger
        ) { attempt in
            // Re-check per attempt, matching HFTreeLister.fetch: flipping
            // offlineMode mid-operation stops the walk at the next request.
            guard !HFClient.offlineMode else {
                throw DownloadError.networkDisabled(operation: "download(\(path))")
            }
            var attemptRequest = request
            var resumeOffset: Int64 = 0

            let partialSize = fileSize(at: partialURL)
            let validator = (try? String(contentsOf: validatorURL, encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if partialSize > 0, let validator, !validator.isEmpty {
                if expectedSize > 0 && partialSize >= Int64(expectedSize) {
                    // A previous run finished the body but didn't get to move
                    // the file. Validate and reuse it instead of re-fetching.
                    if (try? validateDownloadedArtifact(
                        at: partialURL, response: nil, path: path, expectedSize: expectedSize))
                        != nil
                    {
                        try? FileManager.default.removeItem(at: validatorURL)
                        return partialURL
                    }
                    clearPartialDownload(partialURL)
                } else {
                    attemptRequest.setValue("bytes=\(partialSize)-", forHTTPHeaderField: "Range")
                    attemptRequest.setValue(validator, forHTTPHeaderField: "If-Range")
                    resumeOffset = partialSize
                    logger.info(
                        "Resuming \(path) from byte \(partialSize) (attempt \(attempt))")
                }
            } else if partialSize > 0 {
                // Partial bytes with no validator can't be safely resumed.
                clearPartialDownload(partialURL)
            }

            let httpResponse = try await streamDownload(
                request: attemptRequest,
                to: partialURL,
                resumeOffset: resumeOffset,
                configuration: configuration,
                onProgress: onProgress,
                onResponse: { response in
                    // Persist the validator as soon as the fresh body starts, so
                    // a drop mid-body still leaves it for the next attempt. A 206
                    // keeps the validator of the response it is extending.
                    guard response.statusCode != 206, (200..<300).contains(response.statusCode)
                    else { return }
                    if let validator = resumeValidator(from: response) {
                        try? validator.write(to: validatorURL, atomically: true, encoding: .utf8)
                    } else {
                        try? FileManager.default.removeItem(at: validatorURL)
                    }
                }
            )

            try HFClient.checkRateLimitForRetry(httpResponse, context: "downloading \(path)")

            if httpResponse.statusCode == 416 {
                // Range not satisfiable — the partial is bogus (or the remote
                // shrank). Restart from 0 on the next attempt.
                clearPartialDownload(partialURL)
                throw DownloadError.invalidArtifact(
                    path: path, reason: "server rejected resume range (416)")
            }

            guard (200..<300).contains(httpResponse.statusCode) else {
                throw DownloadError.downloadFailed(
                    path: path,
                    underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                )
            }

            // Validate before the caller moves the file into the cache.
            do {
                try validateDownloadedArtifact(
                    at: partialURL, response: httpResponse, path: path,
                    expectedSize: expectedSize)
            } catch {
                // A completed-but-invalid body must not be resumed from.
                clearPartialDownload(partialURL)
                throw error
            }

            try? FileManager.default.removeItem(at: validatorURL)
            return partialURL
        }
    }

    private static func downloadInParallelRanges(
        request: URLRequest,
        path: String,
        expectedSize: Int,
        partialURL: URL,
        validatorURL: URL,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        maxAttempts: Int,
        minBackoff: TimeInterval,
        configuration: URLSessionConfiguration?,
        rangeConfiguration: ParallelRangeConfiguration
    ) async throws -> URL {
        guard !HFClient.offlineMode else {
            throw DownloadError.networkDisabled(operation: "download(\(path))")
        }

        let ranges = makeParallelRanges(
            totalBytes: expectedSize, chunkSize: rangeConfiguration.chunkSize)
        let rangeDirectory = partialURL.appendingPathExtension("ranges")
        try? FileManager.default.removeItem(at: rangeDirectory)
        try FileManager.default.createDirectory(at: rangeDirectory, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: validatorURL)
        defer { try? FileManager.default.removeItem(at: rangeDirectory) }

        logger.info(
            "Starting parallel range download for \(path): \(ranges.count) ranges, "
                + "max concurrent \(rangeConfiguration.maxConcurrent)")
        let progressState = ParallelRangeState(
            rangeCount: ranges.count,
            expectedBytes: Int64(expectedSize),
            progress: onProgress
        )

        let failures: [ParallelRangeTaskOutcome] = await withTaskGroup(
            of: ParallelRangeTaskOutcome.self
        ) { group in
            var failures: [ParallelRangeTaskOutcome] = []
            var nextIndex = 0
            var stopped = false

            func scheduleNext() {
                guard !stopped, nextIndex < ranges.count else { return }
                let range = ranges[nextIndex]
                nextIndex += 1
                group.addTask {
                    do {
                        try await downloadParallelRange(
                            range,
                            request: request,
                            path: path,
                            expectedSize: expectedSize,
                            rangeDirectory: rangeDirectory,
                            progressState: progressState,
                            maxAttempts: maxAttempts,
                            minBackoff: minBackoff,
                            configuration: configuration
                        )
                        return ParallelRangeTaskOutcome(index: range.index, error: nil)
                    } catch {
                        return ParallelRangeTaskOutcome(index: range.index, error: error)
                    }
                }
            }

            for _ in 0..<min(rangeConfiguration.maxConcurrent, ranges.count) {
                scheduleNext()
            }

            while let outcome = await group.next() {
                if outcome.error != nil {
                    failures.append(outcome)
                    if !stopped {
                        stopped = true
                        group.cancelAll()
                    }
                }
                scheduleNext()
            }
            return failures
        }

        if Task.isCancelled {
            persistParallelRangePrefix(
                ranges: ranges,
                rangeDirectory: rangeDirectory,
                partialURL: partialURL,
                validatorURL: validatorURL,
                validator: progressState.acceptedValidator(),
                path: path
            )
            try Task.checkCancellation()
        }

        if !failures.isEmpty {
            let ordered = failures.sorted { $0.index < $1.index }
            let selected = ordered.first { outcome in
                guard let error = outcome.error else { return false }
                return !RetryPolicy.isCancellation(error)
            } ?? ordered[0]
            let error = selected.error!
            if !(error is ParallelRangeError) {
                persistParallelRangePrefix(
                    ranges: ranges,
                    rangeDirectory: rangeDirectory,
                    partialURL: partialURL,
                    validatorURL: validatorURL,
                    validator: progressState.acceptedValidator(),
                    path: path
                )
            }
            throw error
        }

        let assembledBytes = try assembleParallelRangePrefix(
            ranges: ranges, rangeDirectory: rangeDirectory, partialURL: partialURL)
        guard assembledBytes == Int64(expectedSize) else {
            throw DownloadError.invalidArtifact(
                path: path,
                reason: "parallel assembly wrote \(assembledBytes) of \(expectedSize) bytes")
        }

        try validateDownloadedArtifact(
            at: partialURL, response: nil, path: path, expectedSize: expectedSize)
        onProgress?(Int64(expectedSize), Int64(expectedSize))
        logger.info("Finished parallel range download for \(path)")
        return partialURL
    }

    private static func persistParallelRangePrefix(
        ranges: [ParallelRange],
        rangeDirectory: URL,
        partialURL: URL,
        validatorURL: URL,
        validator: String?,
        path: String
    ) {
        guard let validator else { return }
        do {
            let bytes = try assembleParallelRangePrefix(
                ranges: ranges, rangeDirectory: rangeDirectory, partialURL: partialURL)
            guard bytes > 0 else {
                try? FileManager.default.removeItem(at: partialURL)
                return
            }
            try validator.write(to: validatorURL, atomically: true, encoding: .utf8)
            logger.info("Preserved \(bytes) resumable bytes after parallel download stopped: \(path)")
        } catch {
            clearPartialDownload(partialURL)
            logger.warning(
                "Could not preserve parallel download progress for \(path): \(error.localizedDescription)")
        }
    }

    @discardableResult
    private static func assembleParallelRangePrefix(
        ranges: [ParallelRange],
        rangeDirectory: URL,
        partialURL: URL
    ) throws -> Int64 {
        if FileManager.default.fileExists(atPath: partialURL.path) {
            try FileManager.default.removeItem(at: partialURL)
        }
        FileManager.default.createFile(atPath: partialURL.path, contents: nil)
        let output = try FileHandle(forWritingTo: partialURL)
        defer { try? output.close() }

        var assembledBytes: Int64 = 0
        for range in ranges {
            let rangeURL = rangeDirectory.appendingPathComponent("\(range.index).part")
            let availableBytes = min(fileSize(at: rangeURL), Int64(range.byteCount))
            guard availableBytes > 0 else { break }

            let input = try FileHandle(forReadingFrom: rangeURL)
            var remainingBytes = availableBytes
            while remainingBytes > 0 {
                let readCount = min(Int64(1 * 1024 * 1024), remainingBytes)
                let data = try input.read(upToCount: Int(readCount)) ?? Data()
                guard !data.isEmpty else { break }
                try output.write(contentsOf: data)
                assembledBytes += Int64(data.count)
                remainingBytes -= Int64(data.count)
            }
            try input.close()
            guard availableBytes == Int64(range.byteCount) else { break }
        }
        try output.synchronize()
        return assembledBytes
    }

    private static func downloadParallelRange(
        _ range: ParallelRange,
        request: URLRequest,
        path: String,
        expectedSize: Int,
        rangeDirectory: URL,
        progressState: ParallelRangeState,
        maxAttempts: Int,
        minBackoff: TimeInterval,
        configuration: URLSessionConfiguration?
    ) async throws {
        var rangeRequest = request
        rangeRequest.setValue("bytes=\(range.start)-\(range.end)", forHTTPHeaderField: "Range")
        let destination = rangeDirectory.appendingPathComponent("\(range.index).part")

        _ = try await RetryPolicy.withRetry(
            label: "\(path)#range-\(range.index)",
            maxAttempts: maxAttempts,
            minBackoff: minBackoff,
            logger: logger
        ) { _ in
            try? FileManager.default.removeItem(at: destination)
            let response = try await streamDownload(
                request: rangeRequest,
                to: destination,
                configuration: configuration,
                onProgress: { bytesWritten, _ in
                    progressState.record(range: range, bytesWritten: bytesWritten)
                },
                validateResponse: { response in
                    try validateParallelRangeResponse(
                        response,
                        range: range,
                        path: path,
                        expectedSize: expectedSize,
                        progressState: progressState
                    )
                }
            )
            try validateDownloadedArtifact(
                at: destination,
                response: response,
                path: "\(path)#range-\(range.index)",
                expectedSize: range.byteCount
            )
            progressState.record(range: range, bytesWritten: Int64(range.byteCount))
            return response
        }
    }

    private static func validateParallelRangeResponse(
        _ response: HTTPURLResponse,
        range: ParallelRange,
        path: String,
        expectedSize: Int,
        progressState: ParallelRangeState
    ) throws {
        try HFClient.checkRateLimitForRetry(response, context: "downloading \(path)")
        guard response.statusCode == 206 else {
            if response.statusCode == 200 {
                throw ParallelRangeError.unsupported("server ignored the Range request")
            }
            throw DownloadError.downloadFailed(
                path: path,
                underlying: NSError(domain: "HTTP", code: response.statusCode)
            )
        }

        guard
            let contentRange = response.value(forHTTPHeaderField: "Content-Range"),
            contentRange == "bytes \(range.start)-\(range.end)/\(expectedSize)"
        else {
            throw DownloadError.invalidArtifact(
                path: path, reason: "invalid Content-Range for parallel download")
        }
        guard
            let validator = response.value(forHTTPHeaderField: "ETag"),
            validator.count >= 2,
            validator.first == "\"",
            validator.last == "\"",
            !validator.hasPrefix("W/")
        else {
            throw ParallelRangeError.unsupported("server did not provide a strong ETag")
        }
        try progressState.accept(validator: validator, path: path)
    }

    private static func makeParallelRanges(totalBytes: Int, chunkSize: Int) -> [ParallelRange] {
        var ranges: [ParallelRange] = []
        var start = 0
        while start < totalBytes {
            let end = min(start + chunkSize - 1, totalBytes - 1)
            ranges.append(ParallelRange(index: ranges.count, start: start, end: end))
            start = end + 1
        }
        return ranges
    }
}

// MARK: - URLSession download delegate for byte-level progress

/// `URLSessionDownloadTask` → async/await bridge that forwards byte progress (#756).
/// Streams response bytes straight into a destination file so partial content
/// survives connection drops (#757). Appends after `resumeOffset` on HTTP 206;
/// truncates and writes from 0 on any other 2xx; discards the body otherwise.
private final class StreamingDownloadDelegate: NSObject, URLSessionDataDelegate, Sendable {
    private static let writeBufferSize = 1 * 1024 * 1024

    private struct State {
        var continuation: CheckedContinuation<HTTPURLResponse, Error>?
        var task: URLSessionDataTask?
        var handle: FileHandle?
        var pendingData = Data()
        var bytesWritten: Int64 = 0
        var expectedTotal: Int64 = -1
        var writeError: Error?
        var finished = false
    }

    private let destination: URL
    private let resumeOffset: Int64
    private let onProgress: (@Sendable (Int64, Int64) -> Void)?
    private let onResponse: (@Sendable (HTTPURLResponse) -> Void)?
    private let validateResponse: (@Sendable (HTTPURLResponse) throws -> Void)?
    // Holds the non-Sendable continuation and file handle; the lock (not
    // `@unchecked`) makes the delegate Sendable as URLSession requires.
    private let state = OSAllocatedUnfairLock<State>(uncheckedState: State())

    init(
        destination: URL,
        resumeOffset: Int64,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        onResponse: (@Sendable (HTTPURLResponse) -> Void)?,
        validateResponse: (@Sendable (HTTPURLResponse) throws -> Void)?
    ) {
        self.destination = destination
        self.resumeOffset = resumeOffset
        self.onProgress = onProgress
        self.onResponse = onResponse
        self.validateResponse = validateResponse
    }

    /// Register the continuation and task before the task starts.
    func attach(
        continuation: CheckedContinuation<HTTPURLResponse, Error>,
        task: URLSessionDataTask
    ) {
        state.withLockUnchecked {
            $0.continuation = continuation
            $0.task = task
        }
    }

    func cancel() {
        state.withLockUnchecked { $0.task }?.cancel()
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        do {
            try validateResponse?(http)
        } catch {
            state.withLockUnchecked { $0.writeError = error }
            completionHandler(.cancel)
            return
        }
        onResponse?(http)

        // Only 2xx bodies are file content worth keeping; error bodies are
        // discarded so a 503 page can never overwrite resumable bytes.
        if (200..<300).contains(http.statusCode) {
            let appending = http.statusCode == 206
            do {
                if !FileManager.default.fileExists(atPath: destination.path) {
                    FileManager.default.createFile(atPath: destination.path, contents: nil)
                }
                let handle = try FileHandle(forWritingTo: destination)
                if appending {
                    try handle.seekToEnd()
                } else {
                    try handle.truncate(atOffset: 0)
                }
                let base = appending ? resumeOffset : 0
                // Content-Range carries the full object size on a 206; a plain
                // 200 reports the full size via expectedContentLength.
                let expectedTotal: Int64
                if let contentRange = http.value(forHTTPHeaderField: "Content-Range"),
                    let totalPart = contentRange.split(separator: "/").last,
                    let total = Int64(totalPart)
                {
                    expectedTotal = total
                } else if http.expectedContentLength >= 0 {
                    expectedTotal = base + http.expectedContentLength
                } else {
                    expectedTotal = -1
                }
                state.withLockUnchecked {
                    $0.handle = handle
                    $0.pendingData.removeAll(keepingCapacity: true)
                    $0.bytesWritten = base
                    $0.expectedTotal = expectedTotal
                }
            } catch {
                state.withLockUnchecked { $0.writeError = error }
                completionHandler(.cancel)
                return
            }
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let progress: (written: Int64, expected: Int64)? = state.withLockUnchecked { st in
            guard let handle = st.handle, st.writeError == nil else { return nil }
            do {
                st.pendingData.append(data)
                guard st.pendingData.count >= Self.writeBufferSize else { return nil }
                try handle.write(contentsOf: st.pendingData)
                st.bytesWritten += Int64(st.pendingData.count)
                st.pendingData.removeAll(keepingCapacity: true)
                return (st.bytesWritten, st.expectedTotal)
            } catch {
                st.writeError = error
                return nil
            }
        }
        if let progress {
            onProgress?(progress.written, progress.expected)
        } else if state.withLockUnchecked({ $0.writeError }) != nil {
            cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let response = task.response as? HTTPURLResponse
        // Extract the resume action under the lock, run it after releasing.
        let completion = state.withLockUnchecked {
            st -> (progress: (written: Int64, expected: Int64)?, resume: (() -> Void)?)? in
            guard !st.finished, let continuation = st.continuation else { return nil }
            st.finished = true
            st.continuation = nil
            st.task = nil
            var finalProgress: (written: Int64, expected: Int64)?
            if st.writeError == nil, let handle = st.handle, !st.pendingData.isEmpty {
                do {
                    try handle.write(contentsOf: st.pendingData)
                    st.bytesWritten += Int64(st.pendingData.count)
                    st.pendingData.removeAll(keepingCapacity: true)
                    finalProgress = (st.bytesWritten, st.expectedTotal)
                } catch {
                    st.writeError = error
                }
            }
            try? st.handle?.close()
            st.handle = nil
            let writeError = st.writeError
            let resume = {
                if let writeError {
                    continuation.resume(throwing: writeError)
                } else if let error {
                    continuation.resume(throwing: error)
                } else if let response {
                    continuation.resume(returning: response)
                } else {
                    continuation.resume(throwing: DownloadError.invalidResponse)
                }
            }
            return (finalProgress, resume)
        }
        if let progress = completion?.progress {
            onProgress?(progress.written, progress.expected)
        }
        completion?.resume?()
    }
}
