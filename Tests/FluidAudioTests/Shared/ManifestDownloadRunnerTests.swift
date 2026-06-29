import Foundation
import os.lock
import XCTest

@testable import FluidAudio

final class ManifestDownloadRunnerTests: XCTestCase {
    override func tearDown() {
        DownloadTestURLProtocol.reset()
        super.tearDown()
    }

    func testAggregatesProgressAcrossMultipleUnits() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        DownloadTestURLProtocol.enqueue(path: "a.bin", statusCode: 200, chunks: [Data("abcd".utf8)])
        DownloadTestURLProtocol.enqueue(path: "b.bin", statusCode: 200, chunks: [Data("efghij".utf8)])

        let unitA = ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "a.bin", localRelativePath: "a.bin", sizeBytes: 4)
            ]),
            destinationRoot: tempDir.appendingPathComponent("RepoA"),
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )
        let unitB = ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "b.bin", localRelativePath: "b.bin", sizeBytes: 6)
            ]),
            destinationRoot: tempDir.appendingPathComponent("RepoB"),
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )

        let runner = ManifestDownloadRunner(downloader: makeDownloader())
        let samples = DownloadProgressRecorder()
        try await runner.run(units: [unitA, unitB], legacyFractionMultiplier: 1.0) {
            samples.append($0)
        }
        let recordedSamples = samples.values

        XCTAssertEqual(recordedSamples.last?.downloadedBytes, 10)
        XCTAssertEqual(recordedSamples.last?.totalBytes, 10)
        XCTAssertEqual(recordedSamples.last?.downloadFractionCompleted, 1.0)
        XCTAssertEqual(try Data(contentsOf: tempDir.appendingPathComponent("RepoA/a.bin")), Data("abcd".utf8))
        XCTAssertEqual(try Data(contentsOf: tempDir.appendingPathComponent("RepoB/b.bin")), Data("efghij".utf8))
    }

    func testMixedManifestStartsFromExistingBytesAndFinishesMonotonically() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let existing = tempDir.appendingPathComponent("existing.bin")
        try Data("done".utf8).write(to: existing)

        let partialDestination = tempDir.appendingPathComponent("partial.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: partialDestination)
        try Data("ab".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(
            path: "partial.bin",
            statusCode: 206,
            headers: ["Content-Range": "bytes 2-4/5"],
            chunks: [Data("cde".utf8)]
        )
        DownloadTestURLProtocol.enqueue(
            path: "missing.bin",
            statusCode: 200,
            chunks: [Data("xyz".utf8)]
        )

        let unit = ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "existing.bin", localRelativePath: "existing.bin", sizeBytes: 4),
                DownloadFile(remotePath: "partial.bin", localRelativePath: "partial.bin", sizeBytes: 5),
                DownloadFile(remotePath: "missing.bin", localRelativePath: "missing.bin", sizeBytes: 3),
                DownloadFile(remotePath: "empty.bin", localRelativePath: "empty.bin", sizeBytes: 0),
            ]),
            destinationRoot: tempDir,
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )

        let samples = DownloadProgressRecorder()
        try await ManifestDownloadRunner(downloader: makeDownloader()).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) {
            samples.append($0)
        }
        let recordedSamples = samples.values
        let byteSamples = recordedSamples.compactMap(\.downloadedBytes)

        XCTAssertEqual(recordedSamples.first?.downloadedBytes, 4)
        XCTAssertEqual(recordedSamples.first?.totalBytes, 12)
        XCTAssertEqual(recordedSamples.last?.downloadedBytes, 12)
        XCTAssertEqual(recordedSamples.last?.downloadFractionCompleted, 1.0)
        XCTAssertEqual(byteSamples, byteSamples.sorted())
        XCTAssertEqual(try Data(contentsOf: partialDestination), Data("abcde".utf8))
        XCTAssertEqual(try Data(contentsOf: tempDir.appendingPathComponent("missing.bin")), Data("xyz".utf8))
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempDir.appendingPathComponent("empty.bin").path))
        XCTAssertTrue(
            DownloadTestURLProtocol.requests.contains {
                $0.value(forHTTPHeaderField: "Range") == "bytes=2-"
            }
        )
    }

    func testUnknownSizeKeepsByteFractionNilUntilComplete() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        DownloadTestURLProtocol.enqueue(path: "known.bin", statusCode: 200, chunks: [Data("abc".utf8)])
        DownloadTestURLProtocol.enqueue(path: "unknown.bin", statusCode: 200, chunks: [Data("defg".utf8)])

        let unit = ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "known.bin", localRelativePath: "known.bin", sizeBytes: 3),
                DownloadFile(remotePath: "unknown.bin", localRelativePath: "unknown.bin", sizeBytes: nil),
            ]),
            destinationRoot: tempDir,
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )

        let samples = DownloadProgressRecorder()
        try await ManifestDownloadRunner(downloader: makeDownloader()).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) {
            samples.append($0)
        }
        let recordedSamples = samples.values

        XCTAssertTrue(recordedSamples.allSatisfy { $0.totalBytes == nil })
        XCTAssertTrue(recordedSamples.allSatisfy { $0.downloadFractionCompleted == nil })
        XCTAssertEqual(recordedSamples.last?.downloadedBytes, 7)
    }

    func testCorrectSizeFinalFileIsSkippedAndOversizedFinalFileIsRedownloaded() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let correct = tempDir.appendingPathComponent("correct.bin")
        let oversized = tempDir.appendingPathComponent("oversized.bin")
        try Data("good".utf8).write(to: correct)
        try Data("too-large".utf8).write(to: oversized)

        DownloadTestURLProtocol.enqueue(statusCode: 200, chunks: [Data("right".utf8)])

        let unit = ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "correct.bin", localRelativePath: "correct.bin", sizeBytes: 4),
                DownloadFile(remotePath: "oversized.bin", localRelativePath: "oversized.bin", sizeBytes: 5),
            ]),
            destinationRoot: tempDir,
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )

        let samples = DownloadProgressRecorder()
        try await ManifestDownloadRunner(downloader: makeDownloader()).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) {
            samples.append($0)
        }

        XCTAssertEqual(DownloadTestURLProtocol.requests.count, 1)
        XCTAssertEqual(try Data(contentsOf: correct), Data("good".utf8))
        XCTAssertEqual(try Data(contentsOf: oversized), Data("right".utf8))
        XCTAssertEqual(samples.values.first?.downloadedBytes, 4)
        XCTAssertEqual(samples.values.last?.downloadedBytes, 9)
    }

    func testCorruptFinalFileIsRedownloadedBeforeCountingComplete() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        try Data("bad".utf8).write(to: destination)
        DownloadTestURLProtocol.enqueue(statusCode: 200, chunks: [Data("correct".utf8)])

        let unit = ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "file.bin", localRelativePath: "file.bin", sizeBytes: 7)
            ]),
            destinationRoot: tempDir,
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )

        let samples = DownloadProgressRecorder()
        try await ManifestDownloadRunner(downloader: makeDownloader()).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) {
            samples.append($0)
        }
        let recordedSamples = samples.values

        XCTAssertEqual(try Data(contentsOf: destination), Data("correct".utf8))
        XCTAssertEqual(recordedSamples.last?.downloadedBytes, 7)
        XCTAssertEqual(recordedSamples.last?.downloadFractionCompleted, 1.0)
    }

    func testIndependentFilesRespectGlobalConcurrencyCap() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let counter = ActiveDownloadCounter()
        let fileCount = 10

        let unit = ManifestDownloadUnit(
            manifest: DownloadManifest(
                files: (0..<fileCount).map { index in
                    DownloadFile(
                        remotePath: "file-\(index).bin",
                        localRelativePath: "file-\(index).bin",
                        sizeBytes: 1024
                    )
                }),
            destinationRoot: tempDir,
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )

        try await ManifestDownloadRunner(downloadFile: { _, destinationURL, expectedBytes, progress in
            counter.begin()
            defer { counter.end() }
            try await Task.sleep(nanoseconds: 50_000_000)
            let bytes = expectedBytes ?? 1024
            progress(bytes)
            return ValidatedDownloadResult(finalURL: destinationURL, bytesWritten: bytes, resumed: false)
        }).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) { _ in }

        XCTAssertGreaterThanOrEqual(counter.maxActive, 2)
        XCTAssertLessThanOrEqual(counter.maxActive, 6)
    }

    func testRetryableFileDownloadFailureIsRetriedBeforeSuccess() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let unit = ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "file.bin", localRelativePath: "file.bin", sizeBytes: 4)
            ]),
            destinationRoot: tempDir,
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )

        let samples = DownloadProgressRecorder()
        let delays = DoubleRecorder()
        try await ManifestDownloadRunner(
            downloadFile: { _, destinationURL, _, _ in
                let attempt = attempts.increment()
                if attempt < 3 {
                    throw URLError(.networkConnectionLost)
                }
                let data = Data("done".utf8)
                try data.write(to: destinationURL)
                return ValidatedDownloadResult(
                    finalURL: destinationURL, bytesWritten: Int64(data.count), resumed: false)
            },
            retrySleep: { delays.append($0) }
        ).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) {
            samples.append($0)
        }

        XCTAssertEqual(attempts.value, 3)
        XCTAssertEqual(delays.values, [1.0, 2.0])
        XCTAssertEqual(try Data(contentsOf: tempDir.appendingPathComponent("file.bin")), Data("done".utf8))
        XCTAssertEqual(samples.values.last?.downloadedBytes, 4)
        XCTAssertEqual(samples.values.last?.downloadFractionCompleted, 1.0)
    }

    func testRateLimitFailureIsRetriedBeforeSuccess() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let delays = DoubleRecorder()
        let unit = singleFileUnit(destinationRoot: tempDir)
        let samples = DownloadProgressRecorder()

        try await ManifestDownloadRunner(
            downloadFile: { _, destinationURL, _, _ in
                let attempt = attempts.increment()
                if attempt == 1 {
                    throw DownloadUtils.HuggingFaceDownloadError.rateLimited(
                        statusCode: 503,
                        message: "HTTP 503"
                    )
                }
                let data = Data("done".utf8)
                try data.write(to: destinationURL)
                return ValidatedDownloadResult(
                    finalURL: destinationURL, bytesWritten: Int64(data.count), resumed: false)
            },
            retrySleep: { delays.append($0) }
        ).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) {
            samples.append($0)
        }

        XCTAssertEqual(attempts.value, 2)
        XCTAssertEqual(delays.values, [1.0])
        XCTAssertEqual(samples.values.last?.downloadedBytes, 4)
        XCTAssertEqual(try Data(contentsOf: tempDir.appendingPathComponent("file.bin")), Data("done".utf8))
    }

    func testHTTPServerFailureIsRetriedBeforeSuccess() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let delays = DoubleRecorder()
        let unit = singleFileUnit(destinationRoot: tempDir)

        try await ManifestDownloadRunner(
            downloadFile: { _, destinationURL, _, _ in
                let attempt = attempts.increment()
                if attempt == 1 {
                    throw DownloadUtils.HuggingFaceDownloadError.downloadFailed(
                        path: "file.bin",
                        underlying: NSError(domain: "HTTP", code: 500)
                    )
                }
                let data = Data("done".utf8)
                try data.write(to: destinationURL)
                return ValidatedDownloadResult(
                    finalURL: destinationURL, bytesWritten: Int64(data.count), resumed: false)
            },
            retrySleep: { delays.append($0) }
        ).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) { _ in }

        XCTAssertEqual(attempts.value, 2)
        XCTAssertEqual(delays.values, [1.0])
        XCTAssertEqual(try Data(contentsOf: tempDir.appendingPathComponent("file.bin")), Data("done".utf8))
    }

    func testByteCountingShortReadFailureIsRetriedBeforeSuccess() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let delays = DoubleRecorder()
        let unit = singleFileUnit(destinationRoot: tempDir)

        try await ManifestDownloadRunner(
            downloadFile: { _, destinationURL, _, _ in
                let attempt = attempts.increment()
                if attempt == 1 {
                    throw DownloadUtils.HuggingFaceDownloadError.downloadFailed(
                        path: "file.bin",
                        underlying: NSError(
                            domain: "FluidAudio.ByteCountingFileDownloader",
                            code: 1
                        )
                    )
                }
                let data = Data("done".utf8)
                try data.write(to: destinationURL)
                return ValidatedDownloadResult(
                    finalURL: destinationURL, bytesWritten: Int64(data.count), resumed: false)
            },
            retrySleep: { delays.append($0) }
        ).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) { _ in }

        XCTAssertEqual(attempts.value, 2)
        XCTAssertEqual(delays.values, [1.0])
        XCTAssertEqual(try Data(contentsOf: tempDir.appendingPathComponent("file.bin")), Data("done".utf8))
    }


    func testHTTPNotFoundFailureIsNotRetried() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let delays = DoubleRecorder()
        let unit = singleFileUnit(destinationRoot: tempDir)

        do {
            try await ManifestDownloadRunner(
                downloadFile: { _, _, _, _ in
                    attempts.increment()
                    throw DownloadUtils.HuggingFaceDownloadError.downloadFailed(
                        path: "file.bin",
                        underlying: NSError(domain: "HTTP", code: 404)
                    )
                },
                retrySleep: { delays.append($0) }
            ).run(
                units: [unit],
                legacyFractionMultiplier: 1.0
            ) { _ in }
            XCTFail("Expected HTTP 404 to fail without retry")
        } catch DownloadUtils.HuggingFaceDownloadError.downloadFailed(_, let underlying) {
            XCTAssertEqual((underlying as NSError).code, 404)
        }

        XCTAssertEqual(attempts.value, 1)
        XCTAssertEqual(delays.values, [])
    }

    func testRetryableFailureStopsAfterMaxAttempts() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let delays = DoubleRecorder()
        let unit = singleFileUnit(destinationRoot: tempDir)

        do {
            try await ManifestDownloadRunner(
                downloadFile: { _, _, _, _ in
                    attempts.increment()
                    throw URLError(.timedOut)
                },
                retrySleep: { delays.append($0) }
            ).run(
                units: [unit],
                legacyFractionMultiplier: 1.0
            ) { _ in }
            XCTFail("Expected retryable error to fail after max attempts")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .timedOut)
        }

        XCTAssertEqual(attempts.value, 4)
        XCTAssertEqual(delays.values, [1.0, 2.0, 4.0])
    }

    func testRetryProgressReflectsRestartedAttemptWhenRetryReportsFewerBytes() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let unit = singleFileUnit(destinationRoot: tempDir)
        let samples = DownloadProgressRecorder()

        try await ManifestDownloadRunner(
            downloadFile: { _, destinationURL, _, progress in
                let attempt = attempts.increment()
                if attempt == 1 {
                    progress(3)
                    throw URLError(.networkConnectionLost)
                }
                progress(1)
                progress(4)
                let data = Data("done".utf8)
                try data.write(to: destinationURL)
                return ValidatedDownloadResult(
                    finalURL: destinationURL, bytesWritten: Int64(data.count), resumed: false)
            },
            retrySleep: { _ in }
        ).run(
            units: [unit],
            legacyFractionMultiplier: 1.0
        ) {
            samples.append($0)
        }

        let byteSamples = samples.values.compactMap(\.downloadedBytes)
        XCTAssertEqual(attempts.value, 2)
        let failedAttemptIndex = try XCTUnwrap(byteSamples.firstIndex(of: 3))
        let restartedAttemptIndex = try XCTUnwrap(
            byteSamples[(failedAttemptIndex + 1)...].firstIndex(of: 1)
        )
        XCTAssertGreaterThan(restartedAttemptIndex, failedAttemptIndex)
        XCTAssertEqual(byteSamples.last, 4)
    }

    func testNonHTTPDownloadFailureIsNotRetried() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let attempts = IntCounter()
        let unit = singleFileUnit(destinationRoot: tempDir)

        do {
            try await ManifestDownloadRunner(
                downloadFile: { _, _, _, _ in
                    attempts.increment()
                    throw DownloadUtils.HuggingFaceDownloadError.downloadFailed(
                        path: "file.bin",
                        underlying: NSError(domain: NSCocoaErrorDomain, code: NSFileWriteNoPermissionError)
                    )
                },
                retrySleep: { _ in }
            ).run(
                units: [unit],
                legacyFractionMultiplier: 1.0
            ) { _ in }
            XCTFail("Expected local file-system shaped failure to fail without retry")
        } catch DownloadUtils.HuggingFaceDownloadError.downloadFailed(_, let underlying) {
            XCTAssertEqual((underlying as NSError).domain, NSCocoaErrorDomain)
        }

        XCTAssertEqual(attempts.value, 1)
    }

    func testExistingProgressInitializerRemainsSourceCompatible() {
        let progress = DownloadUtils.DownloadProgress(
            fractionCompleted: 0.25,
            phase: .downloading(completedFiles: 1, totalFiles: 4)
        )

        XCTAssertEqual(progress.fractionCompleted, 0.25)
        XCTAssertNil(progress.downloadedBytes)
        XCTAssertNil(progress.totalBytes)
        XCTAssertNil(progress.downloadFractionCompleted)
        XCTAssertNil(progress.currentFile)
    }

    private func singleFileUnit(destinationRoot: URL) -> ManifestDownloadUnit {
        ManifestDownloadUnit(
            manifest: DownloadManifest(files: [
                DownloadFile(remotePath: "file.bin", localRelativePath: "file.bin", sizeBytes: 4)
            ]),
            destinationRoot: destinationRoot,
            makeRequest: { file in URLRequest(url: URL(string: "https://test.local/\(file.remotePath)")!) }
        )
    }

    private func makeDownloader() -> ByteCountingFileDownloader {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [DownloadTestURLProtocol.self]
        return ByteCountingFileDownloader(session: URLSession(configuration: configuration))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ManifestDownloadRunnerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

final class DownloadProgressRecorder: Sendable {
    private let storage = OSAllocatedUnfairLock(initialState: [DownloadUtils.DownloadProgress]())

    var values: [DownloadUtils.DownloadProgress] {
        storage.withLock { $0 }
    }

    func append(_ value: DownloadUtils.DownloadProgress) {
        storage.withLock { $0.append(value) }
    }
}

private final class IntCounter: Sendable {
    private let storage = OSAllocatedUnfairLock<Int>(initialState: 0)

    var value: Int {
        storage.withLock { $0 }
    }

    @discardableResult
    func increment() -> Int {
        storage.withLock { value in
            value += 1
            return value
        }
    }
}

private final class DoubleRecorder: Sendable {
    private let storage = OSAllocatedUnfairLock<[Double]>(initialState: [])

    var values: [Double] {
        storage.withLock { $0 }
    }

    func append(_ value: Double) {
        storage.withLock { $0.append(value) }
    }
}

struct DownloadTestResponse {
    let statusCode: Int
    let headers: [String: String]
    let chunks: [Data]
    let errorAfterChunks: Error?
}

final class DownloadTestURLProtocol: URLProtocol {
    private static let state = OSAllocatedUnfairLock(initialState: DownloadTestURLProtocolState())

    static var requests: [URLRequest] {
        state.withLock { $0.capturedRequests }
    }

    static func enqueue(
        path: String? = nil,
        statusCode: Int,
        headers: [String: String] = [:],
        chunks: [Data] = [],
        errorAfterChunks: Error? = nil
    ) {
        let response = DownloadTestResponse(
            statusCode: statusCode,
            headers: headers,
            chunks: chunks,
            errorAfterChunks: errorAfterChunks
        )
        state.withLock { state in
            if let path {
                state.responsesByPath[path] = response
            } else {
                state.queuedResponses.append(response)
            }
        }
    }

    static func reset() {
        state.withLock { state in
            state.queuedResponses = []
            state.responsesByPath = [:]
            state.capturedRequests = []
        }
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        let request = request
        let response: DownloadTestResponse
        response = Self.state.withLock { state in
            state.capturedRequests.append(request)
            if let key = request.url?.lastPathComponent,
                let response = state.responsesByPath.removeValue(forKey: key)
            {
                return response
            }
            return state.queuedResponses.removeFirst()
        }

        let httpResponse = HTTPURLResponse(
            url: request.url!,
            statusCode: response.statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: response.headers
        )!
        client?.urlProtocol(self, didReceive: httpResponse, cacheStoragePolicy: .notAllowed)

        for chunk in response.chunks {
            client?.urlProtocol(self, didLoad: chunk)
            Thread.sleep(forTimeInterval: 0.01)
        }

        if let error = response.errorAfterChunks {
            client?.urlProtocol(self, didFailWithError: error)
        } else {
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}

private struct DownloadTestURLProtocolState {
    var queuedResponses: [DownloadTestResponse] = []
    var responsesByPath: [String: DownloadTestResponse] = [:]
    var capturedRequests: [URLRequest] = []
}

private final class ActiveDownloadCounter: Sendable {
    private struct State {
        var active = 0
        var maxActive = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var maxActive: Int {
        state.withLock { $0.maxActive }
    }

    func begin() {
        state.withLock { state in
            state.active += 1
            state.maxActive = max(state.maxActive, state.active)
        }
    }

    func end() {
        state.withLock { state in
            state.active -= 1
        }
    }
}
