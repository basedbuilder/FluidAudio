import Foundation
import Network
import os.lock
import XCTest

@testable import FluidAudio

final class ByteCountingFileDownloaderTests: XCTestCase {
    override func tearDown() {
        DownloadTestURLProtocol.reset()
        super.tearDown()
    }

    func testEmitsIncreasingByteProgressBeforeCompletion() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let firstChunk = Data(repeating: 1, count: 1_048_576)
        let secondChunk = Data(repeating: 2, count: 1_048_576)
        let thirdChunk = Data(repeating: 3, count: 1_048_576)
        let totalBytes = Int64(firstChunk.count + secondChunk.count + thirdChunk.count)
        let server = try ChunkedHTTPTestServer(
            statusCode: 200,
            headers: [:],
            chunks: [firstChunk, secondChunk, thirdChunk]
        )
        defer { server.stop() }

        let samples = Int64ProgressRecorder()
        let result = try await makePlainDownloader().download(
            request: URLRequest(url: server.url),
            to: destination,
            expectedBytes: totalBytes,
            progress: { samples.append($0) }
        )
        let recordedSamples = samples.values

        XCTAssertEqual(result.bytesWritten, totalBytes)
        XCTAssertEqual(try ByteCountingFileDownloader.fileSize(at: destination), totalBytes)
        XCTAssertTrue(
            recordedSamples.contains { $0 > 0 && $0 < totalBytes },
            "expected a progress sample before completion, got \(recordedSamples)"
        )
        XCTAssertEqual(recordedSamples.last, totalBytes)
    }

    func testLargeFileDoesNotWaitFor16MBSegmentBeforeProgress() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("large-file.bin")
        let chunkSize = 1_048_576
        let chunkCount = 65
        let body = Data(repeating: 9, count: chunkSize * chunkCount)
        let totalBytes = Int64(body.count)
        let server = try RangeAwareHTTPTestServer(body: body, chunkSize: chunkSize)
        defer { server.stop() }

        let samples = Int64ProgressRecorder()
        let result = try await makePlainDownloader().download(
            request: URLRequest(url: server.url),
            to: destination,
            expectedBytes: totalBytes,
            progress: { samples.append($0) }
        )
        let recordedSamples = samples.values

        XCTAssertEqual(result.bytesWritten, totalBytes)
        XCTAssertEqual(try ByteCountingFileDownloader.fileSize(at: destination), totalBytes)
        XCTAssertTrue(server.observedRangeHeaders.contains("bytes=0-16777215"))
        XCTAssertTrue(
            recordedSamples.contains { $0 > 0 && $0 < 16 * 1_048_576 },
            "expected progress before a 16 MB segment boundary, got \(recordedSamples.prefix(8))"
        )
        XCTAssertEqual(recordedSamples.last, totalBytes)
    }

    func testParallelRangeFallbackKeepsProgressMonotonicAndResetsToFullDownload() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("large-file.bin")
        let chunkSize = 1_048_576
        let chunkCount = 65
        let body = Data(repeating: 6, count: chunkSize * chunkCount)
        let totalBytes = Int64(body.count)
        let server = try RangeAwareHTTPTestServer(
            body: body,
            chunkSize: chunkSize,
            failingRangeStarts: [16 * 1_048_576],
            failureDelayMilliseconds: 80
        )
        defer { server.stop() }

        let samples = Int64ProgressRecorder()
        let result = try await makePlainDownloader().download(
            request: URLRequest(url: server.url),
            to: destination,
            expectedBytes: totalBytes,
            progress: { samples.append($0) }
        )
        let recordedSamples = samples.values

        XCTAssertEqual(result.bytesWritten, totalBytes)
        XCTAssertEqual(try Data(contentsOf: destination), body)
        XCTAssertTrue(server.observedRangeHeaders.contains("bytes=0-16777215"))
        XCTAssertTrue(server.observedRangeHeaders.contains("bytes=16777216-33554431"))
        XCTAssertTrue(server.observedRangeHeaders.contains(nil))
        XCTAssertTrue(
            isNonDecreasing(recordedSamples),
            "expected monotonic progress, got \(recordedSamples)"
        )
        XCTAssertTrue(
            recordedSamples.contains { $0 > 0 && $0 < totalBytes },
            "expected intermediate progress, got \(recordedSamples)"
        )
        XCTAssertEqual(recordedSamples.last, totalBytes)
    }

    func testResumesPartialFileWithRangeRequest() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abc".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(
            statusCode: 206,
            headers: ["Content-Range": "bytes 3-5/6"],
            chunks: [Data("def".utf8)]
        )

        let samples = Int64ProgressRecorder()
        let result = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: 6,
            progress: { samples.append($0) }
        )
        let recordedSamples = samples.values

        XCTAssertTrue(result.resumed)
        XCTAssertEqual(DownloadTestURLProtocol.requests.first?.value(forHTTPHeaderField: "Range"), "bytes=3-")
        XCTAssertEqual(try Data(contentsOf: destination), Data("abcdef".utf8))
        XCTAssertEqual(recordedSamples.last, 6)
    }

    func testLiveHuggingFaceFullFileReportsThroughputWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment["FLUIDAUDIO_LIVE_DOWNLOAD_SMOKE"] == "1" else {
            throw XCTSkip(
                "set FLUIDAUDIO_LIVE_DOWNLOAD_SMOKE=1 to run the live Hugging Face full-file smoke test"
            )
        }

        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let expectedBytes: Int64 = 215_951_360
        let destination = tempDir.appendingPathComponent("weight-segmented.bin")
        let url = URL(
            string:
                "https://huggingface.co/FluidInference/parakeet-tdt-ctc-110m-coreml/resolve/main/Preprocessor.mlmodelc/weights/weight.bin"
        )!

        let samples = Int64ProgressRecorder()
        let start = Date()
        let result = try await ByteCountingFileDownloader().download(
            request: URLRequest(url: url, timeoutInterval: 120),
            to: destination,
            expectedBytes: expectedBytes,
            progress: { samples.append($0) }
        )
        let elapsed = Date().timeIntervalSince(start)
        let megabitsPerSecond = (Double(expectedBytes) * 8.0 / 1_000_000.0) / elapsed
        print(
            "FLUIDAUDIO_LIVE_DOWNLOAD_SMOKE bytes=\(expectedBytes) elapsed_s=\(elapsed) mbit_s=\(megabitsPerSecond)"
        )

        XCTAssertEqual(result.bytesWritten, expectedBytes)
        XCTAssertEqual(try ByteCountingFileDownloader.fileSize(at: destination), expectedBytes)
        XCTAssertGreaterThanOrEqual(samples.values.filter { $0 > 0 && $0 < expectedBytes }.count, 4)
    }

    func testRestartsWhenServerIgnoresRangeRequest() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abc".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(
            statusCode: 200,
            headers: ["Content-Length": "6"],
            chunks: [Data("abcdef".utf8)]
        )

        _ = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: 6,
            progress: { _ in }
        )

        XCTAssertEqual(try Data(contentsOf: destination), Data("abcdef".utf8))
    }

    func testHTTPErrorDoesNotAppendBodyToPartialFile() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abc".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(
            statusCode: 404,
            chunks: [Data("not a model".utf8)]
        )

        do {
            _ = try await makeDownloader().download(
                request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
                to: destination,
                expectedBytes: 6,
                progress: { _ in }
            )
            XCTFail("expected HTTP failure")
        } catch {
            XCTAssertEqual(try Data(contentsOf: partial), Data("abc".utf8))
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        }
    }

    func testRateLimitErrorsDoNotAppendBodyToPartialFile() async throws {
        for statusCode in [429, 503] {
            DownloadTestURLProtocol.reset()
            let tempDir = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: tempDir) }

            let destination = tempDir.appendingPathComponent("file-\(statusCode).bin")
            let partial = ByteCountingFileDownloader.partialURL(for: destination)
            try Data("abc".utf8).write(to: partial)

            DownloadTestURLProtocol.enqueue(
                statusCode: statusCode,
                chunks: [Data("retry later".utf8)]
            )

            do {
                _ = try await makeDownloader().download(
                    request: URLRequest(url: URL(string: "https://test.local/file-\(statusCode).bin")!),
                    to: destination,
                    expectedBytes: 6,
                    progress: { _ in }
                )
                XCTFail("expected rate limit failure for \(statusCode)")
            } catch {
                XCTAssertEqual(try Data(contentsOf: partial), Data("abc".utf8))
                XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
            }
        }
    }

    func testInvalidContentRangeRestartsWithoutAppendingDuplicateBytes() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abc".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(
            statusCode: 206,
            headers: ["Content-Range": "bytes 0-5/6"],
            chunks: [Data("abcdef".utf8)]
        )
        DownloadTestURLProtocol.enqueue(
            statusCode: 200,
            headers: ["Content-Length": "6"],
            chunks: [Data("abcdef".utf8)]
        )

        _ = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: 6,
            progress: { _ in }
        )

        XCTAssertEqual(DownloadTestURLProtocol.requests.count, 2)
        XCTAssertEqual(try Data(contentsOf: destination), Data("abcdef".utf8))
    }

    func testMissingContentRangeRestartsWithoutAppendingDuplicateBytes() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abc".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(
            statusCode: 206,
            chunks: [Data("def".utf8)]
        )
        DownloadTestURLProtocol.enqueue(
            statusCode: 200,
            headers: ["Content-Length": "6"],
            chunks: [Data("abcdef".utf8)]
        )

        _ = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: 6,
            progress: { _ in }
        )

        XCTAssertEqual(DownloadTestURLProtocol.requests.count, 2)
        XCTAssertEqual(try Data(contentsOf: destination), Data("abcdef".utf8))
    }

    func testMismatchedContentRangeTotalRestartsWithoutAppendingDuplicateBytes() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abc".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(
            statusCode: 206,
            headers: ["Content-Range": "bytes 3-5/7"],
            chunks: [Data("def".utf8)]
        )
        DownloadTestURLProtocol.enqueue(
            statusCode: 200,
            headers: ["Content-Length": "6"],
            chunks: [Data("abcdef".utf8)]
        )

        _ = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: 6,
            progress: { _ in }
        )

        XCTAssertEqual(DownloadTestURLProtocol.requests.count, 2)
        XCTAssertEqual(try Data(contentsOf: destination), Data("abcdef".utf8))
    }

    func testRangeNotSatisfiableRestartsStalePartial() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abc".utf8).write(to: partial)

        DownloadTestURLProtocol.enqueue(statusCode: 416)
        DownloadTestURLProtocol.enqueue(
            statusCode: 200,
            headers: ["Content-Length": "6"],
            chunks: [Data("abcdef".utf8)]
        )

        _ = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: 6,
            progress: { _ in }
        )

        XCTAssertEqual(DownloadTestURLProtocol.requests.count, 2)
        XCTAssertEqual(try Data(contentsOf: destination), Data("abcdef".utf8))
    }

    func testCompletePartialFileIsFinalizedWithoutNetworkRequest() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)
        try Data("abcdef".utf8).write(to: partial)

        let result = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: 6,
            progress: { _ in }
        )

        XCTAssertTrue(result.resumed)
        XCTAssertEqual(DownloadTestURLProtocol.requests.count, 0)
        XCTAssertEqual(try Data(contentsOf: destination), Data("abcdef".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
    }

    func testAInterruptedTransferLeavesReusablePartialFile() async throws {
        let tempDir = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("file.bin")
        let partial = ByteCountingFileDownloader.partialURL(for: destination)

        let prefix = Data(repeating: 7, count: 1_048_576)
        let suffix = Data(repeating: 8, count: 1_048_576)
        let expectedBytes = Int64(prefix.count + suffix.count)
        let server = try ChunkedHTTPTestServer(
            statusCode: 200,
            chunks: [prefix],
            contentLength: Int(expectedBytes),
            closeAfterChunks: true
        )
        defer { server.stop() }

        do {
            _ = try await makePlainDownloader().download(
                request: URLRequest(url: server.url),
                to: destination,
                expectedBytes: expectedBytes,
                progress: { _ in }
            )
            XCTFail("expected interrupted transfer failure")
        } catch {
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
            XCTAssertEqual(try waitForFileSize(at: partial), Int64(prefix.count))
        }

        DownloadTestURLProtocol.enqueue(
            statusCode: 206,
            headers: ["Content-Range": "bytes \(prefix.count)-\(prefix.count + suffix.count - 1)/\(expectedBytes)"],
            chunks: [suffix]
        )

        _ = try await makeDownloader().download(
            request: URLRequest(url: URL(string: "https://test.local/file.bin")!),
            to: destination,
            expectedBytes: expectedBytes,
            progress: { _ in }
        )

        XCTAssertEqual(
            DownloadTestURLProtocol.requests.last?.value(forHTTPHeaderField: "Range"), "bytes=\(prefix.count)-")
        XCTAssertEqual(try ByteCountingFileDownloader.fileSize(at: destination), expectedBytes)
    }

    private func makeDownloader() -> ByteCountingFileDownloader {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [DownloadTestURLProtocol.self]
        return ByteCountingFileDownloader(session: URLSession(configuration: configuration))
    }

    private func makePlainDownloader() -> ByteCountingFileDownloader {
        ByteCountingFileDownloader(session: URLSession(configuration: .ephemeral))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ByteCountingFileDownloaderTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func waitForFileSize(at url: URL) throws -> Int64 {
        for _ in 0..<20 {
            if FileManager.default.fileExists(atPath: url.path) {
                return try ByteCountingFileDownloader.fileSize(at: url)
            }
            Thread.sleep(forTimeInterval: 0.025)
        }
        return try ByteCountingFileDownloader.fileSize(at: url)
    }

    private func isNonDecreasing(_ values: [Int64]) -> Bool {
        guard values.count > 1 else { return true }
        for index in 1..<values.count {
            if values[index] < values[index - 1] {
                return false
            }
        }
        return true
    }
}

final class ChunkedHTTPTestServer: Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "ChunkedHTTPTestServer")
    private let chunks: [Data]
    private let statusCode: Int
    private let headers: [String: String]
    private let contentLength: Int?
    private let closeAfterChunks: Bool

    var url: URL {
        URL(string: "http://127.0.0.1:\(listener.port!.rawValue)/file.bin")!
    }

    init(
        statusCode: Int,
        headers: [String: String] = [:],
        chunks: [Data],
        contentLength: Int? = nil,
        closeAfterChunks: Bool = false
    ) throws {
        self.listener = try NWListener(using: .tcp, on: .any)
        self.chunks = chunks
        self.statusCode = statusCode
        self.headers = headers
        self.contentLength = contentLength
        self.closeAfterChunks = closeAfterChunks

        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            if case .ready = state {
                ready.signal()
            }
        }
        listener.newConnectionHandler = { [self] connection in
            handle(connection: connection)
        }
        listener.start(queue: queue)
        _ = ready.wait(timeout: .now() + 2)
    }

    func stop() {
        listener.cancel()
        Thread.sleep(forTimeInterval: 0.1)
    }

    private func handle(connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [self] _, _, _, _ in
            sendResponse(on: connection)
        }
    }

    private func sendResponse(on connection: NWConnection) {
        let bodyLength = contentLength ?? chunks.reduce(0) { $0 + $1.count }
        var headerFields = headers
        headerFields["Content-Length"] = "\(bodyLength)"
        headerFields["Connection"] = "close"

        let statusText = HTTPURLResponse.localizedString(forStatusCode: statusCode)
        var response = "HTTP/1.1 \(statusCode) \(statusText)\r\n"
        for (key, value) in headerFields {
            response += "\(key): \(value)\r\n"
        }
        response += "\r\n"

        connection.send(
            content: Data(response.utf8),
            completion: .contentProcessed { [self] _ in
                sendChunk(at: 0, on: connection)
            })
    }

    private func sendChunk(at index: Int, on connection: NWConnection) {
        guard index < chunks.count else {
            if closeAfterChunks {
                connection.cancel()
            } else {
                connection.send(content: nil, completion: .contentProcessed { _ in connection.cancel() })
            }
            return
        }

        connection.send(
            content: chunks[index],
            completion: .contentProcessed { [self] _ in
                queue.asyncAfter(deadline: .now() + .milliseconds(25)) {
                    self.sendChunk(at: index + 1, on: connection)
                }
            })
    }
}

final class RangeAwareHTTPTestServer: Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "RangeAwareHTTPTestServer")
    private let body: Data
    private let chunkSize: Int
    private let failingRangeStarts: Set<Int>
    private let failureDelayMilliseconds: Int
    private let firstRangeHeaderStorage = OSAllocatedUnfairLock<String?>(initialState: nil)
    private let rangeHeaderStorage = OSAllocatedUnfairLock<[String?]>(initialState: [])

    var url: URL {
        URL(string: "http://127.0.0.1:\(listener.port!.rawValue)/large-file.bin")!
    }

    var firstRangeHeader: String? {
        firstRangeHeaderStorage.withLock { $0 }
    }

    var observedRangeHeaders: [String?] {
        rangeHeaderStorage.withLock { $0 }
    }

    init(
        body: Data,
        chunkSize: Int,
        failingRangeStarts: Set<Int> = [],
        failureDelayMilliseconds: Int = 0
    ) throws {
        self.listener = try NWListener(using: .tcp, on: .any)
        self.body = body
        self.chunkSize = chunkSize
        self.failingRangeStarts = failingRangeStarts
        self.failureDelayMilliseconds = failureDelayMilliseconds

        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            if case .ready = state {
                ready.signal()
            }
        }
        listener.newConnectionHandler = { [self] connection in
            handle(connection: connection)
        }
        listener.start(queue: queue)
        _ = ready.wait(timeout: .now() + 2)
    }

    func stop() {
        listener.cancel()
        Thread.sleep(forTimeInterval: 0.1)
    }

    private func handle(connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [self] data, _, _, _ in
            let rangeHeader = parseRangeHeader(from: data ?? Data())
            firstRangeHeaderStorage.withLock { stored in
                if stored == nil {
                    stored = rangeHeader
                }
            }
            rangeHeaderStorage.withLock { $0.append(rangeHeader) }
            sendResponse(on: connection, rangeHeader: rangeHeader)
        }
    }

    private func parseRangeHeader(from data: Data) -> String? {
        guard let request = String(data: data, encoding: .utf8) else { return nil }
        for line in request.components(separatedBy: "\r\n") {
            guard line.lowercased().hasPrefix("range:") else { continue }
            return line.dropFirst("range:".count).trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    private func sendResponse(on connection: NWConnection, rangeHeader: String?) {
        let requestedRange = parseRange(rangeHeader)
        let start = requestedRange?.start ?? 0
        let end = requestedRange?.end ?? body.count - 1
        if failingRangeStarts.contains(start) {
            sendFailure(on: connection)
            return
        }
        let statusCode = requestedRange == nil ? 200 : 206
        let responseBody = body.subdata(in: start..<(end + 1))
        var headers = [
            "Content-Length": "\(responseBody.count)",
            "Connection": "close",
        ]
        if statusCode == 206 {
            headers["Content-Range"] = "bytes \(start)-\(end)/\(body.count)"
        }

        let statusText = HTTPURLResponse.localizedString(forStatusCode: statusCode)
        var response = "HTTP/1.1 \(statusCode) \(statusText)\r\n"
        for (key, value) in headers {
            response += "\(key): \(value)\r\n"
        }
        response += "\r\n"

        connection.send(
            content: Data(response.utf8),
            completion: .contentProcessed { [self] _ in
                sendChunk(from: responseBody, offset: 0, on: connection)
            })
    }

    private func sendFailure(on connection: NWConnection) {
        let response = """
            HTTP/1.1 500 internal server error\r
            Content-Length: 0\r
            Connection: close\r
            \r

            """
        let send = {
            connection.send(
                content: Data(response.utf8),
                completion: .contentProcessed { _ in connection.cancel() }
            )
        }
        if failureDelayMilliseconds > 0 {
            queue.asyncAfter(deadline: .now() + .milliseconds(failureDelayMilliseconds), execute: send)
        } else {
            send()
        }
    }

    private func parseRange(_ rangeHeader: String?) -> (start: Int, end: Int)? {
        guard let rangeHeader, rangeHeader.hasPrefix("bytes=") else { return nil }
        let value = rangeHeader.dropFirst("bytes=".count)
        let bounds = value.split(separator: "-", maxSplits: 1)
        guard
            bounds.count == 2,
            let start = Int(bounds[0]),
            let end = Int(bounds[1]),
            start >= 0,
            end >= start,
            end < body.count
        else {
            return nil
        }
        return (start, end)
    }

    private func sendChunk(from data: Data, offset: Int, on connection: NWConnection) {
        guard offset < data.count else {
            connection.send(content: nil, completion: .contentProcessed { _ in connection.cancel() })
            return
        }

        let end = min(offset + chunkSize, data.count)
        connection.send(
            content: data.subdata(in: offset..<end),
            completion: .contentProcessed { [self] _ in
                queue.asyncAfter(deadline: .now() + .milliseconds(10)) {
                    self.sendChunk(from: data, offset: end, on: connection)
                }
            })
    }
}

final class Int64ProgressRecorder: Sendable {
    private let storage = OSAllocatedUnfairLock(initialState: [Int64]())

    var values: [Int64] {
        storage.withLock { $0 }
    }

    func append(_ value: Int64) {
        storage.withLock { $0.append(value) }
    }
}
