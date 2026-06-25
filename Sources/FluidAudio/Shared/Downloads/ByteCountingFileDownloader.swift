import Foundation
import os.lock

struct ByteCountingFileDownloader {
    private let sessionConfiguration: URLSessionConfiguration

    init(session: URLSession = DownloadUtils.sharedSession) {
        self.sessionConfiguration = session.configuration
    }

    func download(
        request: URLRequest,
        to destinationURL: URL,
        expectedBytes: Int64?,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws -> ValidatedDownloadResult {
        let partialURL = Self.partialURL(for: destinationURL)
        try DownloadUtils.createDirectoryRobustly(at: destinationURL.deletingLastPathComponent())

        if let expectedBytes, FileManager.default.fileExists(atPath: destinationURL.path) {
            let size = try Self.fileSize(at: destinationURL)
            if size == expectedBytes {
                return ValidatedDownloadResult(finalURL: destinationURL, bytesWritten: expectedBytes, resumed: false)
            }
            try FileManager.default.removeItem(at: destinationURL)
        }

        if let expectedBytes, FileManager.default.fileExists(atPath: partialURL.path) {
            let partialSize = try Self.fileSize(at: partialURL)
            if partialSize == expectedBytes {
                try Self.replaceItem(at: destinationURL, with: partialURL)
                return ValidatedDownloadResult(finalURL: destinationURL, bytesWritten: expectedBytes, resumed: true)
            }
            if partialSize > expectedBytes {
                try FileManager.default.removeItem(at: partialURL)
            }
        }

        do {
            return try await performDownload(
                request: request,
                destinationURL: destinationURL,
                partialURL: partialURL,
                expectedBytes: expectedBytes,
                resumeOffset: Self.existingFileSize(at: partialURL),
                progress: progress
            )
        } catch ByteCountingDownloadError.staleRange {
            try? FileManager.default.removeItem(at: partialURL)
            return try await performDownload(
                request: request,
                destinationURL: destinationURL,
                partialURL: partialURL,
                expectedBytes: expectedBytes,
                resumeOffset: 0,
                progress: progress
            )
        }
    }

    private func performDownload(
        request: URLRequest,
        destinationURL: URL,
        partialURL: URL,
        expectedBytes: Int64?,
        resumeOffset: Int64,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws -> ValidatedDownloadResult {
        var request = request
        if resumeOffset > 0 {
            request.setValue("bytes=\(resumeOffset)-", forHTTPHeaderField: "Range")
        }

        return try await withCheckedThrowingContinuation { continuation in
            let queue = OperationQueue()
            queue.maxConcurrentOperationCount = 1
            queue.name = "FluidAudio.ByteCountingFileDownloader"

            let delegate = ByteCountingDataDelegate(
                destinationURL: destinationURL,
                partialURL: partialURL,
                expectedBytes: expectedBytes,
                resumeOffset: resumeOffset,
                progress: progress,
                continuation: continuation
            )
            let session = URLSession(configuration: sessionConfiguration, delegate: delegate, delegateQueue: queue)
            session.dataTask(with: request).resume()
        }
    }

    static func partialURL(for destinationURL: URL) -> URL {
        destinationURL.deletingLastPathComponent()
            .appendingPathComponent(destinationURL.lastPathComponent + ".part")
    }

    fileprivate static func replaceItem(at destinationURL: URL, with sourceURL: URL) throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.moveItem(at: sourceURL, to: destinationURL)
    }

    static func fileSize(at url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static func existingFileSize(at url: URL) -> Int64 {
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
        return (try? fileSize(at: url)) ?? 0
    }
}

private enum ByteCountingDownloadError: Error {
    case staleRange
}

private final class ByteCountingDataDelegate: NSObject, URLSessionDataDelegate {
    private let destinationURL: URL
    private let partialURL: URL
    private let expectedBytes: Int64?
    private let resumeOffset: Int64
    private let progress: @Sendable (Int64) -> Void
    private let continuation: CheckedContinuation<ValidatedDownloadResult, Error>
    private let state: OSAllocatedUnfairLock<ByteCountingDataState>

    init(
        destinationURL: URL,
        partialURL: URL,
        expectedBytes: Int64?,
        resumeOffset: Int64,
        progress: @escaping @Sendable (Int64) -> Void,
        continuation: CheckedContinuation<ValidatedDownloadResult, Error>
    ) {
        self.destinationURL = destinationURL
        self.partialURL = partialURL
        self.expectedBytes = expectedBytes
        self.resumeOffset = resumeOffset
        self.progress = progress
        self.continuation = continuation
        self.state = OSAllocatedUnfairLock(
            initialState:
                ByteCountingDataState(
                    bytesWritten: resumeOffset,
                    resumed: resumeOffset > 0
                ))
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        do {
            guard let response = response as? HTTPURLResponse else {
                throw DownloadUtils.HuggingFaceDownloadError.invalidResponse
            }

            try validate(response: response)
            let bytesWritten = try openPartialFile(for: response)
            progress(bytesWritten)
            completionHandler(.allow)
        } catch {
            completionHandler(.cancel)
            finish(with: .failure(error))
            session.invalidateAndCancel()
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            guard let bytesWritten = try append(data) else { return }
            progress(bytesWritten)
        } catch {
            finish(with: .failure(error))
            session.invalidateAndCancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let snapshot = state.withLock { state in
            (completed: state.completed, bytesWritten: state.bytesWritten, resumed: state.resumed)
        }
        guard !snapshot.completed else { return }

        if let error {
            finish(with: .failure(error))
            session.finishTasksAndInvalidate()
            return
        }

        do {
            if let expectedBytes, snapshot.bytesWritten != expectedBytes {
                throw DownloadUtils.HuggingFaceDownloadError.downloadFailed(
                    path: destinationURL.lastPathComponent,
                    underlying: NSError(
                        domain: "FluidAudio.ByteCountingFileDownloader",
                        code: 1,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Downloaded \(snapshot.bytesWritten) bytes, expected \(expectedBytes)"
                        ]
                    )
                )
            }

            try ByteCountingFileDownloader.replaceItem(at: destinationURL, with: partialURL)
            finish(
                with: .success(
                    ValidatedDownloadResult(
                        finalURL: destinationURL,
                        bytesWritten: snapshot.bytesWritten,
                        resumed: snapshot.resumed
                    )))
        } catch {
            finish(with: .failure(error))
        }
        session.finishTasksAndInvalidate()
    }

    private func validate(response: HTTPURLResponse) throws {
        if response.statusCode == 429 || response.statusCode == 503 {
            throw DownloadUtils.HuggingFaceDownloadError.rateLimited(
                statusCode: response.statusCode,
                message: "HTTP \(response.statusCode)"
            )
        }

        if resumeOffset > 0 {
            switch response.statusCode {
            case 200:
                return
            case 206:
                let contentRange = try ContentRange(response: response)
                guard contentRange.start == resumeOffset else {
                    throw ByteCountingDownloadError.staleRange
                }
                if let expectedBytes, contentRange.total != expectedBytes {
                    throw ByteCountingDownloadError.staleRange
                }
                return
            case 416:
                if let expectedBytes, resumeOffset == expectedBytes {
                    return
                }
                throw ByteCountingDownloadError.staleRange
            default:
                break
            }
        }

        guard (200..<300).contains(response.statusCode) else {
            throw DownloadUtils.HuggingFaceDownloadError.downloadFailed(
                path: destinationURL.lastPathComponent,
                underlying: NSError(domain: "HTTP", code: response.statusCode)
            )
        }
    }

    private func openPartialFile(for response: HTTPURLResponse) throws -> Int64 {
        try state.withLock { state in
            if resumeOffset > 0, response.statusCode == 206 {
                if !FileManager.default.fileExists(atPath: partialURL.path) {
                    FileManager.default.createFile(atPath: partialURL.path, contents: nil)
                }
                let handle = try FileHandle(forWritingTo: partialURL)
                handle.seekToEndOfFile()
                state.fileHandle = handle
                return state.bytesWritten
            }

            if resumeOffset > 0, response.statusCode == 416, let expectedBytes, resumeOffset == expectedBytes {
                state.bytesWritten = expectedBytes
                return state.bytesWritten
            }

            if FileManager.default.fileExists(atPath: partialURL.path) {
                try FileManager.default.removeItem(at: partialURL)
            }
            FileManager.default.createFile(atPath: partialURL.path, contents: nil)
            state.fileHandle = try FileHandle(forWritingTo: partialURL)
            if resumeOffset > 0 {
                state.bytesWritten = 0
                state.resumed = false
            }
            return state.bytesWritten
        }
    }

    private func append(_ data: Data) throws -> Int64? {
        try state.withLock { state in
            guard !state.completed else { return nil }
            if state.fileHandle == nil {
                if !FileManager.default.fileExists(atPath: partialURL.path) {
                    FileManager.default.createFile(atPath: partialURL.path, contents: nil)
                }
                state.fileHandle = try FileHandle(forWritingTo: partialURL)
                state.fileHandle?.seekToEndOfFile()
            }
            state.fileHandle?.write(data)
            try? state.fileHandle?.synchronize()
            state.bytesWritten += Int64(data.count)
            return state.bytesWritten
        }
    }

    private func closeFile(state: inout ByteCountingDataState) {
        try? state.fileHandle?.synchronize()
        try? state.fileHandle?.close()
        state.fileHandle = nil
    }

    private func finish(with result: Result<ValidatedDownloadResult, Error>) {
        let shouldResume = state.withLock { state in
            guard !state.completed else { return false }
            state.completed = true
            closeFile(state: &state)
            return true
        }
        if shouldResume {
            continuation.resume(with: result)
        }
    }
}

private struct ByteCountingDataState {
    var fileHandle: FileHandle?
    var bytesWritten: Int64
    var resumed: Bool
    var completed = false
}

private struct ContentRange {
    let start: Int64
    let total: Int64

    init(response: HTTPURLResponse) throws {
        guard
            let value = response.value(forHTTPHeaderField: "Content-Range"),
            value.hasPrefix("bytes ")
        else {
            throw ByteCountingDownloadError.staleRange
        }

        let rangeAndTotal = value.dropFirst("bytes ".count).split(separator: "/", maxSplits: 1)
        guard rangeAndTotal.count == 2 else {
            throw ByteCountingDownloadError.staleRange
        }

        let bounds = rangeAndTotal[0].split(separator: "-", maxSplits: 1)
        guard
            bounds.count == 2,
            let start = Int64(bounds[0]),
            let total = Int64(rangeAndTotal[1])
        else {
            throw ByteCountingDownloadError.staleRange
        }

        self.start = start
        self.total = total
    }
}
