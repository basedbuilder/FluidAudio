import Foundation
import XCTest
import os

@testable import FluidAudio

/// Tests for HTTP Range resume in `downloadFileWithRetry` (#757): partial
/// bytes survive mid-transfer drops, retries send `Range`/`If-Range`, a 200
/// after a range request replaces (never splices) the partial file, and a
/// fully-downloaded partial from a previous run is reused without network.
final class DownloadResumeTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("DownloadResumeTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        ResumeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        ResumeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    // MARK: - Helpers

    private var stubConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ResumeStubURLProtocol.self]
        return config
    }

    private var request: URLRequest {
        URLRequest(url: URL(string: "https://stub.test/repo/resolve/main/model.bin")!)
    }

    private func partialURL(_ name: String = "model.bin") -> URL {
        workDir.appendingPathComponent(name).appendingPathExtension("partial")
    }

    /// Body content that passes artifact validation (not empty, not HTML).
    private static let fullBody = Data("BINARYCONTENT-0123456789-ABCDEFGHIJ".utf8)

    private func download(
        expectedSize: Int = fullBody.count,
        partial: URL? = nil,
        maxAttempts: Int = 3
    ) async throws -> Data {
        let url = try await FileDownloader.download(
            request: request,
            path: "model.bin",
            expectedSize: expectedSize,
            partialFileURL: partial ?? partialURL(),
            onProgress: nil,
            maxAttempts: maxAttempts,
            minBackoff: 0.01,
            configuration: stubConfiguration
        )
        return try Data(contentsOf: url)
    }

    private func seedPartial(_ data: Data, validator: String?) throws {
        try data.write(to: partialURL())
        if let validator {
            try validator.write(
                to: FileDownloader.resumeValidatorURL(for: partialURL()),
                atomically: true, encoding: .utf8)
        }
    }

    // MARK: - Tests

    func testFreshDownloadWritesFullBody() async throws {
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertNil(requests[0].value(forHTTPHeaderField: "Range"))
        // Validator sidecar is cleaned up after a completed download.
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: FileDownloader.resumeValidatorURL(for: partialURL()).path))
    }

    func testMidStreamDropResumesWithRangeAndAppends() async throws {
        let splitAt = 10
        let head = Self.fullBody.prefix(splitAt)
        let tail = Self.fullBody.dropFirst(splitAt)

        // Attempt 1: 200 with ETag, drops after `head` bytes.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200, headers: ["ETag": "\"v1\""],
                chunks: [Data(head), Data(tail)], failAfterChunks: 1))
        // Attempt 2: a range-capable server. Whether the head bytes survived
        // the drop is racy at the URLProtocol layer (data delivery may be
        // discarded on failure), so the stub answers both shapes: 206 tail if
        // the client resumed, 200 full if it restarted. Either way the final
        // file must be complete and unspliced. The deterministic
        // resume-append path is covered by the seeded-partial tests below.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [], rangeAwareBody: Self.fullBody))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody, "post-drop download must produce the complete file")
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertNil(requests[0].value(forHTTPHeaderField: "Range"))
        if let range = requests[1].value(forHTTPHeaderField: "Range") {
            // Bytes reached disk before the drop → a true resume from whatever
            // offset survived, guarded by the validator captured on attempt 1.
            XCTAssertTrue(range.hasPrefix("bytes="), "got: \(range)")
            XCTAssertEqual(requests[1].value(forHTTPHeaderField: "If-Range"), "\"v1\"")
        }
    }

    func testServerIgnoringRangeReplacesPartialInsteadOfSplicing() async throws {
        // Stale partial from a different file version.
        try seedPartial(Data("STALEBYTES".utf8), validator: "\"v0\"")
        // Server replies 200 (range ignored / file changed) with the new body.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody, "a 200 must replace the partial, never splice")
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=10-")
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "If-Range"), "\"v0\"")
    }

    func testMismatchedResumeContentRangeRetriesFromByteZero() async throws {
        let expectedBody = Data("ABCD".utf8)
        try seedPartial(Data("AB".utf8), validator: "\"v1\"")
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes 0-1/4",
                    "ETag": "\"v1\"",
                ],
                chunks: [Data("AB".utf8)]))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [expectedBody]))

        let body = try await download(expectedSize: expectedBody.count, maxAttempts: 2)

        XCTAssertEqual(body, expectedBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=2-")
        if requests.count == 2 {
            XCTAssertNil(
                requests[1].value(forHTTPHeaderField: "Range"),
                "an invalid resume response must clear the partial and retry from byte zero")
        }
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: FileDownloader.resumeValidatorURL(for: partialURL()).path))
    }

    func testMalformedResumeContentLengthRetriesFromByteZero() async throws {
        let expectedBody = Data("ABCD".utf8)
        try seedPartial(Data("AB".utf8), validator: "\"v1\"")
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes 2-3/4",
                    "Content-Length": "2x",
                    "ETag": "\"v1\"",
                ],
                chunks: [Data("CD".utf8)],
                addsContentLengthAutomatically: false))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [expectedBody]))

        let body = try await download(expectedSize: expectedBody.count, maxAttempts: 2)

        XCTAssertEqual(body, expectedBody)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().count,
            2,
            "a malformed Content-Length must be rejected before appending")
    }

    func testMissingResumeContentLengthRetriesFromByteZero() async throws {
        let expectedBody = Data("ABCD".utf8)
        try seedPartial(Data("AB".utf8), validator: "\"v1\"")
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes 2-3/4",
                    "ETag": "\"v1\"",
                ],
                chunks: [Data("CD".utf8)],
                addsContentLengthAutomatically: false))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [expectedBody]))

        let body = try await download(expectedSize: expectedBody.count, maxAttempts: 2)

        XCTAssertEqual(body, expectedBody)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().count,
            2,
            "a missing Content-Length must be rejected before appending")
    }

    func testUnknownSizeShortResumeRetriesFromByteZero() async throws {
        let expectedBody = Data("ABCDEFGHIJ".utf8)
        try seedPartial(Data("AB".utf8), validator: "\"v1\"")
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes 2-3/10",
                    "ETag": "\"v1\"",
                ],
                chunks: [Data("CD".utf8)]))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [expectedBody]))

        let body = try await download(expectedSize: -1, maxAttempts: 2)

        XCTAssertEqual(body, expectedBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests.first?.value(forHTTPHeaderField: "Range"), "bytes=2-")
        guard requests.count == 2 else { return }
        XCTAssertNil(
            requests[1].value(forHTTPHeaderField: "Range"),
            "an incomplete unknown-size resume must retry from byte zero")
    }

    func testRangeNotSatisfiableClearsPartialAndRetriesFresh() async throws {
        try seedPartial(Data("BOGUSPARTIALBYTES".utf8), validator: "\"v0\"")
        ResumeStubURLProtocol.enqueue(.init(status: 416, headers: [:], chunks: []))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertNotNil(requests[0].value(forHTTPHeaderField: "Range"))
        XCTAssertNil(
            requests[1].value(forHTTPHeaderField: "Range"),
            "after a 416 the partial is cleared and the retry starts from byte 0")
    }

    func testNoValidatorMeansNoResume() async throws {
        let splitAt = 10
        // Attempt 1: 200 with only a weak ETag (unusable for If-Range), drops mid-body.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200, headers: ["ETag": "W/\"weak\""],
                chunks: [Data(Self.fullBody.prefix(splitAt)), Data(Self.fullBody.dropFirst(splitAt))],
                failAfterChunks: 1))
        // Attempt 2 must be a fresh full download.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: [:], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertNil(
            requests[1].value(forHTTPHeaderField: "Range"),
            "without a strong validator the partial must be discarded, not resumed")
    }

    func testErrorBodyDoesNotClobberPartialBytes() async throws {
        let splitAt = 10
        try seedPartial(Data(Self.fullBody.prefix(splitAt)), validator: "\"v1\"")
        // Attempt 1: 503 with an HTML error page body — must not touch the partial.
        ResumeStubURLProtocol.enqueue(
            .init(status: 503, headers: [:], chunks: [Data("<html>err</html>".utf8)]))
        // Attempt 2: resume succeeds.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes \(splitAt)-\(Self.fullBody.count - 1)/\(Self.fullBody.count)"
                ],
                chunks: [Data(Self.fullBody.dropFirst(splitAt))]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests[1].value(forHTTPHeaderField: "Range"), "bytes=\(splitAt)-")
    }

    func testCompletedPartialFromPreviousRunIsReusedWithoutNetwork() async throws {
        try seedPartial(Self.fullBody, validator: "\"v1\"")
        // No scripts enqueued — any network request would fail the test.

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        XCTAssertTrue(
            ResumeStubURLProtocol.recordedRequests().isEmpty,
            "a complete, valid partial must be reused without hitting the network")
    }

    // MARK: - Stall watchdog (#810)

    /// A frozen mid-transfer connection is cancelled by the watchdog and the
    /// retry resumes to completion — instead of hanging on the idle `timeout`.
    func testStalledTransferIsCancelledAndRetried() async throws {
        // Attempt 1: 200 + ETag, delivers a small head, then freezes. The head
        // is far below the 1 MiB threshold so the watchdog trips on its first
        // (sub-second) window.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"v1\"", "Content-Length": String(Self.fullBody.count)],
                chunks: [Data(Self.fullBody.prefix(10))],
                stall: true))
        // Attempt 2: a range-capable server completes the file.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [], rangeAwareBody: Self.fullBody))

        let fastWatchdog = DownloadConfig(minStallBytes: 1 << 20, stallWindow: 0.2)
        let url = try await FileDownloader.download(
            request: request,
            path: "model.bin",
            expectedSize: Self.fullBody.count,
            partialFileURL: partialURL(),
            onProgress: nil,
            maxAttempts: 3,
            minBackoff: 0.01,
            config: fastWatchdog,
            configuration: stubConfiguration
        )

        XCTAssertEqual(try Data(contentsOf: url), Self.fullBody)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().count, 2,
            "the stalled attempt must be cancelled and retried")
    }

    /// A disabled watchdog (`minStallBytes == 0`) must not cancel a healthy,
    /// progressing download.
    func testDisabledWatchdogDoesNotCancelHealthyDownload() async throws {
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let noWatchdog = DownloadConfig(minStallBytes: 0, stallWindow: 0)
        let url = try await FileDownloader.download(
            request: request,
            path: "model.bin",
            expectedSize: Self.fullBody.count,
            partialFileURL: partialURL(),
            onProgress: nil,
            maxAttempts: 1,
            minBackoff: 0.01,
            config: noWatchdog,
            configuration: stubConfiguration
        )

        XCTAssertEqual(try Data(contentsOf: url), Self.fullBody)
        XCTAssertEqual(ResumeStubURLProtocol.recordedRequests().count, 1)
    }

    func testStalledErrorIsClassifiedRetryable() {
        XCTAssertTrue(RetryPolicy.isRetryable(DownloadError.stalled(path: "model.bin", window: 120)))
        XCTAssertFalse(RetryPolicy.isCancellation(DownloadError.stalled(path: "model.bin", window: 120)))
    }

    func testBufferedNetworkProgressDoesNotTripWatchdogBeforeDurableWrite() async throws {
        let chunkSize = 64 * 1024
        let body = Data(repeating: 0x42, count: chunkSize * 4)
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"v1\""],
                chunks: stride(from: 0, to: body.count, by: chunkSize).map { start in
                    Data(body[start..<min(start + chunkSize, body.count)])
                },
                delayBetweenChunks: 0.12))

        let url = try await FileDownloader.download(
            request: request,
            path: "healthy-buffered.bin",
            expectedSize: body.count,
            partialFileURL: partialURL("healthy-buffered.bin"),
            onProgress: nil,
            maxAttempts: 1,
            minBackoff: 0.01,
            config: DownloadConfig(minStallBytes: 1, stallWindow: 0.2),
            configuration: stubConfiguration
        )

        XCTAssertEqual(try Data(contentsOf: url), body)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().count,
            1,
            "received bytes below the write-buffer boundary must count as network progress")
    }

    func testResumedOffsetDoesNotMaskFirstWindowStall() async throws {
        let prefix = Data(repeating: 0x41, count: 64)
        let expectedSize = 128
        try seedPartial(prefix, validator: "\"v1\"")
        let partial = partialURL()
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Length": "64",
                    "Content-Range": "bytes 64-127/128",
                    "ETag": "\"v1\"",
                ],
                chunks: [],
                stall: true))

        let stalledInFirstWindow = await withTaskGroup(of: Bool.self) { group in
            group.addTask { [request, stubConfiguration] in
                do {
                    _ = try await FileDownloader.download(
                        request: request,
                        path: "resumed-stall.bin",
                        expectedSize: expectedSize,
                        partialFileURL: partial,
                        onProgress: nil,
                        maxAttempts: 1,
                        minBackoff: 0.01,
                        config: DownloadConfig(minStallBytes: 32, stallWindow: 0.5),
                        configuration: stubConfiguration
                    )
                    return false
                } catch DownloadError.stalled {
                    return true
                } catch {
                    return false
                }
            }
            group.addTask {
                try? await Task.sleep(for: .milliseconds(750))
                return false
            }
            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }

        XCTAssertTrue(
            stalledInFirstWindow,
            "the existing 64-byte partial must be the watchdog baseline, not apparent new progress")
    }

    func testParallelRangeReceivesWatchdogConfigAndRetriesStall() async throws {
        let body = Data((0..<128).map { UInt8($0) })
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Length": "64",
                    "Content-Range": "bytes 0-63/128",
                    "ETag": "\"model-v1\"",
                ],
                chunks: [],
                stall: true))
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"model-v1\""],
                chunks: [],
                rangeAwareBody: body))
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"model-v1\""],
                chunks: [],
                rangeAwareBody: body))

        let url = try await FileDownloader.download(
            request: request,
            path: "parallel-stall.bin",
            expectedSize: body.count,
            partialFileURL: partialURL("parallel-stall.bin"),
            onProgress: nil,
            maxAttempts: 2,
            minBackoff: 0.01,
            config: DownloadConfig(minStallBytes: 1, stallWindow: 0.2),
            configuration: stubConfiguration,
            parallelRanges: .init(threshold: 64, chunkSize: 64, maxConcurrent: 1)
        )

        XCTAssertEqual(try Data(contentsOf: url), body)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().compactMap {
                $0.value(forHTTPHeaderField: "Range")
            },
            ["bytes=0-63", "bytes=0-63", "bytes=64-127"])
    }

    func testCallerCancellationWinsAfterWatchdogFires() async throws {
        let bufferedChunk = Data(repeating: 0x42, count: 1 * 1024 * 1024)
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: [
                    "Content-Length": String(bufferedChunk.count * 2),
                    "ETag": "\"v1\"",
                ],
                chunks: [bufferedChunk],
                stall: true))
        let progressGate = BlockingProgressProbe()
        let downloadRequest = request
        let configuration = stubConfiguration
        let destination = partialURL("watchdog-cancel.bin")
        let streamTask = Task {
            try await FileDownloader.streamDownload(
                request: downloadRequest,
                to: destination,
                path: "watchdog-cancel.bin",
                minStallBytes: Int64(bufferedChunk.count * 2),
                stallWindow: 0.2,
                configuration: configuration,
                onProgress: { _, _ in progressGate.record() }
            )
        }

        let progressStarted = await progressGate.waitForFirstCallbackAsync()
        XCTAssertTrue(progressStarted)
        for _ in 0..<200 where ResumeStubURLProtocol.stoppedRequestCount() == 0 {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(
            ResumeStubURLProtocol.stoppedRequestCount(),
            1,
            "the watchdog must cancel the transport before caller cancellation joins the race")
        streamTask.cancel()
        progressGate.releaseFirstCallback()

        do {
            _ = try await streamTask.value
            XCTFail("expected caller cancellation")
        } catch {
            XCTAssertTrue(
                RetryPolicy.isCancellation(error),
                "caller cancellation must win over a watchdog that already marked the task stalled")
        }
    }

    func testFullyReceivedBodyIsNotStalledWhileFinalProgressCallbackRuns() async throws {
        let body = Data(repeating: 0x42, count: 1 * 1024 * 1024)
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [body]))
        let progressGate = BlockingProgressProbe()
        let downloadRequest = request
        let configuration = stubConfiguration
        let destination = partialURL("complete-body.bin")
        let streamTask = Task {
            try await FileDownloader.streamDownload(
                request: downloadRequest,
                to: destination,
                path: "complete-body.bin",
                minStallBytes: Int64(body.count * 2),
                stallWindow: 0.2,
                configuration: configuration,
                onProgress: { _, _ in progressGate.record() }
            )
        }

        let progressStarted = await progressGate.waitForFirstCallbackAsync()
        XCTAssertTrue(progressStarted)
        try await Task.sleep(for: .milliseconds(400))
        progressGate.releaseFirstCallback()

        _ = try await streamTask.value
        XCTAssertEqual(try Data(contentsOf: destination), body)
    }

    func testFullyReceivedParallelRangeIsNotStalledWhileFinalProgressCallbackRuns() async throws {
        let body = Data(repeating: 0x42, count: 64)
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Length": String(body.count),
                    "Content-Range": "bytes 0-63/128",
                    "ETag": "\"v1\"",
                ],
                chunks: [body]))
        let progressGate = BlockingProgressProbe()
        let downloadRequest = request
        let configuration = stubConfiguration
        let destination = partialURL("complete-range.bin")
        let streamTask = Task {
            try await FileDownloader.streamDownload(
                request: downloadRequest,
                to: destination,
                path: "complete-range.bin",
                minStallBytes: 128,
                stallWindow: 0.2,
                configuration: configuration,
                onProgress: { _, _ in progressGate.record() }
            )
        }

        let progressStarted = await progressGate.waitForFirstCallbackAsync()
        XCTAssertTrue(progressStarted)
        try await Task.sleep(for: .milliseconds(400))
        progressGate.releaseFirstCallback()

        _ = try await streamTask.value
        XCTAssertEqual(try Data(contentsOf: destination), body)
    }

    func testCallerCancellationDuringFinalProgressWins() async throws {
        let body = Data(repeating: 0x42, count: 64)
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [body]))
        let progressGate = BlockingProgressProbe()
        let downloadRequest = request
        let configuration = stubConfiguration
        let destination = partialURL("final-progress-cancel.bin")
        let streamTask = Task {
            try await FileDownloader.streamDownload(
                request: downloadRequest,
                to: destination,
                configuration: configuration,
                onProgress: { _, _ in progressGate.record() }
            )
        }

        let progressStarted = await progressGate.waitForFirstCallbackAsync()
        XCTAssertTrue(progressStarted)
        streamTask.cancel()
        progressGate.releaseFirstCallback()

        do {
            _ = try await streamTask.value
            XCTFail("expected cancellation during final progress delivery")
        } catch {
            XCTAssertTrue(RetryPolicy.isCancellation(error), "unexpected error: \(error)")
        }
    }

    func testProgressIncludesResumedOffsetSoItNeverDips() async throws {
        let splitAt = 10
        try seedPartial(Data(Self.fullBody.prefix(splitAt)), validator: "\"v1\"")
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes \(splitAt)-\(Self.fullBody.count - 1)/\(Self.fullBody.count)"
                ],
                chunks: [Data(Self.fullBody.dropFirst(splitAt))]))

        let recorded = ProgressRecorder()
        _ = try await FileDownloader.download(
            request: request,
            path: "model.bin",
            expectedSize: Self.fullBody.count,
            partialFileURL: partialURL(),
            onProgress: { written, expected in recorded.append((written, expected)) },
            maxAttempts: 1,
            minBackoff: 0.01,
            configuration: stubConfiguration
        )

        let events = recorded.snapshot()
        XCTAssertFalse(events.isEmpty)
        XCTAssertGreaterThan(
            events[0].written, Int64(splitAt) - 1,
            "first progress callback must already include the resumed offset")
        XCTAssertEqual(events.last?.written, Int64(Self.fullBody.count))
        XCTAssertEqual(events.last?.expected, Int64(Self.fullBody.count))
    }

    func testProgressReportsDurableWritesAtBufferBoundary() async throws {
        let boundary = 1 * 1024 * 1024
        let body = Data(repeating: 0x42, count: boundary + 17)
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"v1\""],
                chunks: [Data(body.prefix(boundary)), Data(body.suffix(17))],
                delayBetweenChunks: 0.05))

        let recorded = ProgressRecorder()
        let url = try await FileDownloader.download(
            request: request,
            path: "buffered-model.bin",
            expectedSize: body.count,
            partialFileURL: partialURL("buffered-model.bin"),
            onProgress: { written, expected in recorded.append((written, expected)) },
            maxAttempts: 1,
            minBackoff: 0.01,
            configuration: stubConfiguration
        )

        XCTAssertEqual(try Data(contentsOf: url), body)
        let events = recorded.snapshot()
        XCTAssertTrue(
            events.contains { $0.written == boundary },
            "the full write buffer should be durable before EOF")
        XCTAssertEqual(events.last?.written, Int64(body.count))
        XCTAssertEqual(events.last?.expected, Int64(body.count))
    }

    func testLargeFreshDownloadUsesValidatedParallelRanges() async throws {
        let body = Data((0..<97).map { UInt8($0) })
        let expectedRanges = stride(from: 0, to: body.count, by: 16).map { start in
            "bytes=\(start)-\(min(start + 15, body.count - 1))"
        }
        for _ in expectedRanges {
            ResumeStubURLProtocol.enqueue(
                .init(
                    status: 200,
                    headers: ["ETag": "\"model-v1\""],
                    chunks: [],
                    rangeAwareBody: body))
        }

        let recorded = ProgressRecorder()
        let blockingProgress = BlockingProgressProbe()
        let downloadRequest = request
        let destination = partialURL("large-model.bin")
        let configuration = stubConfiguration
        let downloadTask = Task {
            try await FileDownloader.download(
                request: downloadRequest,
                path: "large-model.bin",
                expectedSize: body.count,
                partialFileURL: destination,
                onProgress: { written, expected in
                    recorded.append((written, expected))
                    blockingProgress.record()
                },
                maxAttempts: 1,
                minBackoff: 0.01,
                configuration: configuration,
                parallelRanges: .init(threshold: 64, chunkSize: 16, maxConcurrent: 3)
            )
        }

        XCTAssertEqual(blockingProgress.waitForFirstCallback(), .success)
        for _ in 0..<100 where ResumeStubURLProtocol.finishedRequestCount() < expectedRanges.count {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(
            ResumeStubURLProtocol.finishedRequestCount(),
            expectedRanges.count,
            "a blocked progress callback must not stall range state or network completion"
        )
        blockingProgress.releaseFirstCallback()
        let url = try await downloadTask.value

        XCTAssertEqual(try Data(contentsOf: url), body)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().compactMap {
                $0.value(forHTTPHeaderField: "Range")
            }.sorted(),
            expectedRanges.sorted()
        )
        let events = recorded.snapshot()
        XCTAssertEqual(events.last?.written, Int64(body.count))
        XCTAssertEqual(events.last?.expected, Int64(body.count))
        XCTAssertEqual(events.map(\.written), events.map(\.written).sorted())
    }

    func testParallelRangeUnsupportedFallsBackToResumableSingleStream() async throws {
        let body = Data(repeating: 0x42, count: 97)
        for _ in 0..<2 {
            ResumeStubURLProtocol.enqueue(
                .init(
                    status: 200,
                    headers: ["ETag": "\"model-v1\""],
                    chunks: [body]))
        }

        let url = try await FileDownloader.download(
            request: request,
            path: "range-unsupported.bin",
            expectedSize: body.count,
            partialFileURL: partialURL("range-unsupported.bin"),
            onProgress: nil,
            maxAttempts: 1,
            minBackoff: 0.01,
            configuration: stubConfiguration,
            parallelRanges: .init(threshold: 64, chunkSize: 128, maxConcurrent: 1)
        )

        XCTAssertEqual(try Data(contentsOf: url), body)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=0-96")
        XCTAssertNil(requests[1].value(forHTTPHeaderField: "Range"))
    }

    func testParallelRangesRejectMismatchedValidators() async throws {
        let body = Data((0..<97).map { UInt8($0) })
        for validator in ["\"model-v1\"", "\"model-v2\""] {
            ResumeStubURLProtocol.enqueue(
                .init(
                    status: 200,
                    headers: ["ETag": validator],
                    chunks: [],
                    rangeAwareBody: body))
        }
        let partial = partialURL("mismatched-validator.bin")

        do {
            _ = try await FileDownloader.download(
                request: request,
                path: "mismatched-validator.bin",
                expectedSize: body.count,
                partialFileURL: partial,
                onProgress: nil,
                maxAttempts: 1,
                minBackoff: 0.01,
                configuration: stubConfiguration,
                parallelRanges: .init(threshold: 64, chunkSize: 64, maxConcurrent: 1)
            )
            XCTFail("expected mismatched validators to fail")
        } catch DownloadError.invalidArtifact(_, let reason) {
            XCTAssertEqual(reason, "parallel range validators did not match")
        }

        XCTAssertEqual(try Data(contentsOf: partial), Data(body.prefix(64)))
        XCTAssertEqual(
            try String(
                contentsOf: FileDownloader.resumeValidatorURL(for: partial), encoding: .utf8),
            "\"model-v1\"")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: partial.appendingPathExtension("ranges").path))
    }

    func testLargeExistingPartialUsesNormalResumablePath() async throws {
        let body = Data((0..<97).map { UInt8($0) })
        let partial = partialURL("large-resume.bin")
        try Data(body.prefix(10)).write(to: partial)
        try "\"model-v1\"".write(
            to: FileDownloader.resumeValidatorURL(for: partial),
            atomically: true,
            encoding: .utf8
        )
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"model-v1\""],
                chunks: [],
                rangeAwareBody: body))

        let url = try await FileDownloader.download(
            request: request,
            path: "large-resume.bin",
            expectedSize: body.count,
            partialFileURL: partial,
            onProgress: nil,
            maxAttempts: 1,
            minBackoff: 0.01,
            configuration: stubConfiguration,
            parallelRanges: .init(threshold: 64, chunkSize: 16, maxConcurrent: 3)
        )

        XCTAssertEqual(try Data(contentsOf: url), body)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=10-")
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "If-Range"), "\"model-v1\"")
    }

    func testParallelRangeRetryKeepsExactBoundsAndMonotonicProgress() async throws {
        let body = Data((0..<97).map { UInt8($0) })
        ResumeStubURLProtocol.enqueue(.init(status: 503, headers: [:], chunks: []))
        for _ in 0..<2 {
            ResumeStubURLProtocol.enqueue(
                .init(
                    status: 200,
                    headers: ["ETag": "\"model-v1\""],
                    chunks: [],
                    rangeAwareBody: body))
        }
        let recorded = ProgressRecorder()

        let url = try await FileDownloader.download(
            request: request,
            path: "range-retry.bin",
            expectedSize: body.count,
            partialFileURL: partialURL("range-retry.bin"),
            onProgress: { written, expected in recorded.append((written, expected)) },
            maxAttempts: 2,
            minBackoff: 0.001,
            configuration: stubConfiguration,
            parallelRanges: .init(threshold: 64, chunkSize: 64, maxConcurrent: 1)
        )

        XCTAssertEqual(try Data(contentsOf: url), body)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().compactMap {
                $0.value(forHTTPHeaderField: "Range")
            },
            ["bytes=0-63", "bytes=0-63", "bytes=64-96"]
        )
        let written = recorded.snapshot().map(\.written)
        XCTAssertEqual(written, written.sorted())
    }

    func testInvalidParallelContentRangeDoesNotFallBackToFullDownload() async throws {
        let body = Data(repeating: 0x42, count: 97)
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes 0-95/97",
                    "ETag": "\"model-v1\"",
                ],
                chunks: [body]))

        do {
            _ = try await FileDownloader.download(
                request: request,
                path: "invalid-content-range.bin",
                expectedSize: body.count,
                partialFileURL: partialURL("invalid-content-range.bin"),
                onProgress: nil,
                maxAttempts: 1,
                minBackoff: 0.01,
                configuration: stubConfiguration,
                parallelRanges: .init(threshold: 64, chunkSize: 128, maxConcurrent: 1)
            )
            XCTFail("expected invalid Content-Range to fail")
        } catch DownloadError.invalidArtifact(_, let reason) {
            XCTAssertEqual(reason, "invalid Content-Range for parallel download")
        }

        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=0-96")
    }

    func testMissingStrongETagFallsBackToSingleStream() async throws {
        let body = Data(repeating: 0x42, count: 97)
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: [:], chunks: [], rangeAwareBody: body))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"model-v1\""], chunks: [body]))

        let url = try await FileDownloader.download(
            request: request,
            path: "missing-etag.bin",
            expectedSize: body.count,
            partialFileURL: partialURL("missing-etag.bin"),
            onProgress: nil,
            maxAttempts: 1,
            minBackoff: 0.01,
            configuration: stubConfiguration,
            parallelRanges: .init(threshold: 64, chunkSize: 128, maxConcurrent: 1)
        )

        XCTAssertEqual(try Data(contentsOf: url), body)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=0-96")
        XCTAssertNil(requests[1].value(forHTTPHeaderField: "Range"))
    }

    func testParallelFailurePreservesPrefixForNormalResume() async throws {
        let body = Data((0..<97).map { UInt8($0) })
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"model-v1\""],
                chunks: [],
                rangeAwareBody: body))
        ResumeStubURLProtocol.enqueue(.init(status: 500, headers: [:], chunks: []))
        let partial = partialURL("preserved-prefix.bin")

        do {
            _ = try await FileDownloader.download(
                request: request,
                path: "preserved-prefix.bin",
                expectedSize: body.count,
                partialFileURL: partial,
                onProgress: nil,
                maxAttempts: 1,
                minBackoff: 0.01,
                configuration: stubConfiguration,
                parallelRanges: .init(threshold: 64, chunkSize: 64, maxConcurrent: 1)
            )
            XCTFail("expected range failure")
        } catch DownloadError.downloadFailed {
            // Expected: a non-capability HTTP failure must propagate.
        }

        XCTAssertEqual(try Data(contentsOf: partial), Data(body.prefix(64)))
        XCTAssertEqual(
            try String(
                contentsOf: FileDownloader.resumeValidatorURL(for: partial), encoding: .utf8),
            "\"model-v1\"")
        XCTAssertFalse(
            ResumeStubURLProtocol.recordedRequests().contains {
                $0.value(forHTTPHeaderField: "Range") == nil
            })

        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"model-v1\""],
                chunks: [],
                rangeAwareBody: body))
        let resumedURL = try await FileDownloader.download(
            request: request,
            path: "preserved-prefix.bin",
            expectedSize: body.count,
            partialFileURL: partial,
            onProgress: nil,
            maxAttempts: 1,
            minBackoff: 0.01,
            configuration: stubConfiguration,
            parallelRanges: .init(threshold: 64, chunkSize: 64, maxConcurrent: 1)
        )
        XCTAssertEqual(try Data(contentsOf: resumedURL), body)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().last?.value(forHTTPHeaderField: "Range"),
            "bytes=64-")
    }

    func testParentCancellationWinsAndPreservesCompletedPrefix() async throws {
        let body = Data((0..<97).map { UInt8($0) })
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"model-v1\""],
                chunks: [],
                rangeAwareBody: body))
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"model-v1\""],
                chunks: [],
                responseDelay: 1,
                rangeAwareBody: body))
        let partial = partialURL("cancelled-prefix.bin")
        let downloadRequest = request
        let configuration = stubConfiguration
        let expectedSize = body.count
        let task = Task {
            try await FileDownloader.download(
                request: downloadRequest,
                path: "cancelled-prefix.bin",
                expectedSize: expectedSize,
                partialFileURL: partial,
                onProgress: nil,
                maxAttempts: 1,
                minBackoff: 0.01,
                configuration: configuration,
                parallelRanges: .init(threshold: 64, chunkSize: 64, maxConcurrent: 1)
            )
        }

        for _ in 0..<100 where ResumeStubURLProtocol.recordedRequests().count < 2 {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(ResumeStubURLProtocol.recordedRequests().count, 2)
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch {
            XCTAssertTrue(RetryPolicy.isCancellation(error))
        }
        XCTAssertEqual(try Data(contentsOf: partial), Data(body.prefix(64)))
        XCTAssertEqual(
            try String(
                contentsOf: FileDownloader.resumeValidatorURL(for: partial), encoding: .utf8),
            "\"model-v1\"")
    }

    func testCancellationBeforeTaskAttachmentCancelsStream() async throws {
        let destination = partialURL("cancel-before-attach.bin")
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))
        let gate = CancellationInsensitiveOneShotGate()
        let downloadRequest = request
        let configuration = stubConfiguration
        let streamTask = Task {
            await gate.wait()
            return try await FileDownloader.streamDownload(
                request: downloadRequest,
                to: destination,
                configuration: configuration
            )
        }

        streamTask.cancel()
        await gate.open()

        do {
            _ = try await streamTask.value
            XCTFail("expected cancellation before URL task attachment to win")
        } catch {
            XCTAssertTrue(RetryPolicy.isCancellation(error), "unexpected error: \(error)")
        }
        XCTAssertTrue(
            ResumeStubURLProtocol.recordedRequests().isEmpty,
            "a stream cancelled before task attachment must not start a request")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: destination.path),
            "a stream cancelled before task attachment must not write a partial file")
    }

    func testConcurrentEnsureForSameDestinationUsesOnePartialTransaction() async throws {
        let destination = workDir.appendingPathComponent("concurrent-model.bin")
        let partial = destination.appendingPathExtension("partial")
        let validator = FileDownloader.resumeValidatorURL(for: partial)
        let remoteFile = RemoteFile(path: "concurrent-model.bin", size: Self.fullBody.count)
        let originalBaseURL = ModelRegistry.baseURL
        ModelRegistry.baseURL = "https://stub.test"
        defer { ModelRegistry.baseURL = originalBaseURL }

        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let blockingProgress = BlockingProgressProbe()
        let configuration = stubConfiguration
        let first = Task {
            try await FileDownloader.ensure(
                file: remoteFile,
                from: "repo",
                at: destination,
                recoveringBlockedPaths: false,
                configuration: configuration,
                onBytes: { _, _ in blockingProgress.record() }
            )
        }

        XCTAssertEqual(blockingProgress.waitForFirstCallback(), .success)
        let second = Task {
            try await FileDownloader.ensure(
                file: remoteFile,
                from: "repo",
                at: destination,
                recoveringBlockedPaths: false,
                configuration: configuration
            )
        }

        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().count,
            1,
            "a second ensure for the same destination must wait for the first transaction"
        )
        blockingProgress.releaseFirstCallback()

        let firstOutcome = try await first.value
        let secondOutcome = try await second.value
        guard case .downloaded = firstOutcome else {
            return XCTFail("the transaction owner should download the file")
        }
        guard case .alreadyPresent = secondOutcome else {
            return XCTFail("the waiter should recheck and reuse the completed destination")
        }
        XCTAssertEqual(try Data(contentsOf: destination), Self.fullBody)
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: validator.path))
    }

    func testCancelledSameDestinationWaiterStopsWaitingWithoutCancellingOwner() async throws {
        let destination = workDir.appendingPathComponent("cancelled-waiter.bin")
        let remoteFile = RemoteFile(path: "cancelled-waiter.bin", size: Self.fullBody.count)
        let originalBaseURL = ModelRegistry.baseURL
        ModelRegistry.baseURL = "https://stub.test"
        defer { ModelRegistry.baseURL = originalBaseURL }

        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let blockingProgress = BlockingProgressProbe()
        let configuration = stubConfiguration
        let owner = Task {
            try await FileDownloader.ensure(
                file: remoteFile,
                from: "repo",
                at: destination,
                recoveringBlockedPaths: false,
                configuration: configuration,
                onBytes: { _, _ in blockingProgress.record() }
            )
        }
        XCTAssertEqual(blockingProgress.waitForFirstCallback(), .success)

        let waiter = Task {
            do {
                _ = try await FileDownloader.ensure(
                    file: remoteFile,
                    from: "repo",
                    at: destination,
                    recoveringBlockedPaths: false,
                    configuration: configuration
                )
                return false
            } catch {
                return RetryPolicy.isCancellation(error)
            }
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        waiter.cancel()

        let cancellationObserved = OSAllocatedUnfairLock<Bool?>(initialState: nil)
        let observer = Task {
            let result = await waiter.value
            cancellationObserved.withLock { $0 = result }
        }
        for _ in 0..<100 where cancellationObserved.withLock({ $0 }) == nil {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(
            cancellationObserved.withLock { $0 },
            true,
            "a cancelled waiter must leave the queue before the owner finishes"
        )

        blockingProgress.releaseFirstCallback()
        _ = try await owner.value
        await observer.value
        XCTAssertEqual(try Data(contentsOf: destination), Self.fullBody)
        XCTAssertEqual(ResumeStubURLProtocol.recordedRequests().count, 1)
    }

    func testConcurrentEnsureForDifferentDestinationsDoesNotSerializeGlobally() async throws {
        let firstDestination = workDir.appendingPathComponent("first-model.bin")
        let secondDestination = workDir.appendingPathComponent("second-model.bin")
        let firstRemoteFile = RemoteFile(path: "first-model.bin", size: Self.fullBody.count)
        let secondRemoteFile = RemoteFile(path: "second-model.bin", size: Self.fullBody.count)
        let originalBaseURL = ModelRegistry.baseURL
        ModelRegistry.baseURL = "https://stub.test"
        defer { ModelRegistry.baseURL = originalBaseURL }

        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let blockingProgress = BlockingProgressProbe()
        let configuration = stubConfiguration
        let first = Task {
            try await FileDownloader.ensure(
                file: firstRemoteFile,
                from: "repo",
                at: firstDestination,
                recoveringBlockedPaths: false,
                configuration: configuration,
                onBytes: { _, _ in blockingProgress.record() }
            )
        }
        XCTAssertEqual(blockingProgress.waitForFirstCallback(), .success)

        let second = Task {
            try await FileDownloader.ensure(
                file: secondRemoteFile,
                from: "repo",
                at: secondDestination,
                recoveringBlockedPaths: false,
                configuration: configuration
            )
        }
        for _ in 0..<100 where !FileManager.default.fileExists(atPath: secondDestination.path) {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: secondDestination.path),
            "a different destination must finish while the first destination remains held"
        )
        blockingProgress.releaseFirstCallback()

        _ = try await first.value
        _ = try await second.value
        XCTAssertEqual(try Data(contentsOf: firstDestination), Self.fullBody)
        XCTAssertEqual(try Data(contentsOf: secondDestination), Self.fullBody)
    }
}

// MARK: - Scripted URLProtocol stub

/// Serves a FIFO queue of scripted responses and records every request,
/// including mid-body connection drops (`failAfterChunks`).
final class ResumeStubURLProtocol: URLProtocol {

    struct Script {
        let status: Int
        let headers: [String: String]
        let chunks: [Data]
        var addsContentLengthAutomatically = true
        /// Deliver this many chunks, then fail with `.networkConnectionLost`.
        var failAfterChunks: Int? = nil
        /// Keeps scripted chunks distinct when a test observes streaming behavior.
        var delayBetweenChunks: TimeInterval = 0
        /// Holds the request open so tests can observe concurrency or cancel it.
        var responseDelay: TimeInterval = 0
        /// Deliver the chunks, then go silent forever — never finish, never
        /// fail. Simulates a frozen CDN connection so the stall watchdog fires.
        var stall: Bool = false
        /// When set, respond like a real range-capable server instead of using
        /// `status`/`chunks`: 206 + the slice from the request's `Range` offset
        /// when the header is present, else 200 + the full body. Lets tests
        /// stay deterministic when a prior scripted drop races URLProtocol
        /// data delivery (the partial may or may not contain the bytes).
        var rangeAwareBody: Data? = nil
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var scripts: [Script] = []
    nonisolated(unsafe) private static var requests: [URLRequest] = []
    nonisolated(unsafe) private static var finishedRequests = 0
    nonisolated(unsafe) private static var stoppedRequests = 0

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        scripts = []
        requests = []
        finishedRequests = 0
        stoppedRequests = 0
    }

    static func enqueue(_ script: Script) {
        lock.lock()
        defer { lock.unlock() }
        scripts.append(script)
    }

    static func recordedRequests() -> [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return requests
    }

    static func finishedRequestCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return finishedRequests
    }

    static func stoppedRequestCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return stoppedRequests
    }

    private static func markRequestFinished() {
        lock.lock()
        defer { lock.unlock() }
        finishedRequests += 1
    }

    private static func markRequestStopped() {
        lock.lock()
        defer { lock.unlock() }
        stoppedRequests += 1
    }

    private static func dequeue(recording request: URLRequest) -> Script? {
        lock.lock()
        defer { lock.unlock() }
        requests.append(request)
        return scripts.isEmpty ? nil : scripts.removeFirst()
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let script = Self.dequeue(recording: request), let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        if script.responseDelay > 0 {
            Thread.sleep(forTimeInterval: script.responseDelay)
        }

        if let body = script.rangeAwareBody {
            respondRangeAware(url: url, body: body, headers: script.headers)
            return
        }

        var headers = script.headers
        let bodyBytes = script.chunks.reduce(0) { $0 + $1.count }
        if script.addsContentLengthAutomatically && headers["Content-Length"] == nil {
            headers["Content-Length"] = String(bodyBytes)
        }
        guard
            let response = HTTPURLResponse(
                url: url, statusCode: script.status, httpVersion: "HTTP/1.1",
                headerFields: headers)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        for (index, chunk) in script.chunks.enumerated() {
            if let failAfter = script.failAfterChunks, index >= failAfter {
                failAfterFlush()
                return
            }
            client?.urlProtocol(self, didLoad: chunk)
            if script.delayBetweenChunks > 0, index < script.chunks.count - 1 {
                Thread.sleep(forTimeInterval: script.delayBetweenChunks)
            }
        }
        // Frozen connection: bytes stop flowing but the request stays open, so
        // only the stall watchdog (not the idle timeout) can end it.
        if script.stall { return }
        if let failAfter = script.failAfterChunks, failAfter >= script.chunks.count {
            failAfterFlush()
            return
        }
        client?.urlProtocolDidFinishLoading(self)
        Self.markRequestFinished()
    }

    /// Deliver the scripted connection drop only after already-loaded chunks
    /// have had time to reach the data task's delegate. Failing immediately
    /// after `didLoad` races the URL loading system's internal buffer and can
    /// drop the delivered bytes, which breaks resume tests that rely on the
    /// partial surviving the drop (seen flaky on CI runners).
    private func failAfterFlush() {
        let client = self.client
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) { [self] in
            client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
        }
    }

    /// Behave like a real range-capable server for `body`: 206 + slice when
    /// the request carries `Range: bytes=N-`, else 200 + the full body.
    private func respondRangeAware(url: URL, body: Data, headers: [String: String]) {
        var start = 0
        var end = body.count - 1
        var isRangeRequest = false
        if let range = request.value(forHTTPHeaderField: "Range"), range.hasPrefix("bytes=") {
            let bounds = range.dropFirst("bytes=".count).split(
                separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
            if bounds.count == 2,
                let requestedStart = Int(bounds[0]),
                let requestedEnd = bounds[1].isEmpty ? body.count - 1 : Int(bounds[1]),
                requestedStart >= 0,
                requestedStart <= requestedEnd,
                requestedEnd < body.count
            {
                start = requestedStart
                end = requestedEnd
                isRangeRequest = true
            }
        }

        var responseHeaders = headers
        let payload = body[start...end]
        responseHeaders["Content-Length"] = String(payload.count)
        let status = isRangeRequest ? 206 : 200
        if isRangeRequest {
            responseHeaders["Content-Range"] = "bytes \(start)-\(end)/\(body.count)"
        }
        guard
            let response = HTTPURLResponse(
                url: url, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: responseHeaders)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(payload))
        client?.urlProtocolDidFinishLoading(self)
        Self.markRequestFinished()
    }

    override func stopLoading() {
        Self.markRequestStopped()
    }

}

/// Lock-guarded accumulator for progress events (tests only).
private final class ProgressRecorder: Sendable {
    private let values = OSAllocatedUnfairLock<[(written: Int64, expected: Int64)]>(initialState: [])

    func append(_ value: (written: Int64, expected: Int64)) {
        values.withLock { $0.append(value) }
    }

    func snapshot() -> [(written: Int64, expected: Int64)] {
        values.withLock { $0 }
    }
}

/// Holds the first progress callback while range sessions continue independently.
private final class BlockingProgressProbe: Sendable {
    private let firstCallbackClaimed = OSAllocatedUnfairLock<Bool>(initialState: false)
    private let firstCallbackEntered = DispatchSemaphore(value: 0)
    private let releaseFirst = DispatchSemaphore(value: 0)

    func record() {
        let isFirst = firstCallbackClaimed.withLock { claimed in
            guard !claimed else { return false }
            claimed = true
            return true
        }
        guard isFirst else { return }
        firstCallbackEntered.signal()
        releaseFirst.wait()
    }

    func waitForFirstCallback() -> DispatchTimeoutResult {
        firstCallbackEntered.wait(timeout: .now() + 2)
    }

    func waitForFirstCallbackAsync() async -> Bool {
        for _ in 0..<200 {
            if firstCallbackClaimed.withLock({ $0 }) {
                return true
            }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return firstCallbackClaimed.withLock { $0 }
    }

    func releaseFirstCallback() {
        releaseFirst.signal()
    }
}

/// Holds a task without observing cancellation, so a test can enter the
/// cancellation handler with cancellation already requested.
private actor CancellationInsensitiveOneShotGate {
    private var isOpen = false
    private var waiter: CheckedContinuation<Void, Never>?

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { continuation in
            waiter = continuation
        }
    }

    func open() {
        guard !isOpen else { return }
        isOpen = true
        waiter?.resume()
        waiter = nil
    }
}
