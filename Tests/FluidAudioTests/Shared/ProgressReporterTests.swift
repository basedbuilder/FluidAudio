import XCTest

@testable import FluidAudio

final class ProgressReporterTests: XCTestCase {

    func testDownloadProgressInitializerRemainsSourceCompatible() {
        let progress = DownloadProgress(fractionCompleted: 0.25, phase: .listing)

        XCTAssertNil(progress.downloadedBytes)
        XCTAssertNil(progress.totalBytes)
    }

    func testReporterExposesKnownBytesOnlyDuringDownloads() {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(
            handler: { recorder.append($0) },
            downloadPhaseWeight: 0.5
        )

        reporter.listing()
        reporter.liveBytes(completedBytes: 25, totalBytes: 100, fileIndex: 0, totalFiles: 2)
        reporter.fileBoundary(
            completedBytes: 100,
            totalBytes: 100,
            completedFiles: 2,
            totalFiles: 2
        )
        reporter.compiling(name: "Model", index: 0, count: 1)

        let events = recorder.snapshot()
        XCTAssertNil(events[0].downloadedBytes)
        XCTAssertNil(events[0].totalBytes)
        XCTAssertEqual(events[1].downloadedBytes, 25)
        XCTAssertEqual(events[1].totalBytes, 100)
        XCTAssertEqual(events[2].downloadedBytes, 100)
        XCTAssertEqual(events[2].totalBytes, 100)
        XCTAssertNil(events[3].downloadedBytes)
        XCTAssertNil(events[3].totalBytes)
    }

    func testUnknownByteTotalsRemainNil() {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(
            handler: { recorder.append($0) },
            downloadPhaseWeight: 1.0
        )

        reporter.fileBoundary(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 1,
            totalFiles: 2
        )
        reporter.cachedModelsAvailable()
        reporter.finished()

        for event in recorder.snapshot() {
            XCTAssertNil(event.downloadedBytes)
            XCTAssertNil(event.totalBytes)
        }
    }

    func testMixedSizeManifestReportsKnownByteSubtotalWithoutSignalingCompletion() {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(
            handler: { recorder.append($0) },
            downloadPhaseWeight: 1.0
        )

        reporter.liveBytes(completedBytes: 40, totalBytes: 100, fileIndex: 0, totalFiles: 2)
        reporter.fileBoundary(
            completedBytes: 100,
            totalBytes: 100,
            completedFiles: 1,
            totalFiles: 2
        )
        reporter.fileBoundary(
            completedBytes: 100,
            totalBytes: 100,
            completedFiles: 2,
            totalFiles: 2
        )

        let events = recorder.snapshot()
        XCTAssertEqual(events.map(\.downloadedBytes), [40, 100, 100])
        XCTAssertEqual(events.map(\.totalBytes), [100, 100, 100])
        guard case .downloading(let completed, let total) = events[1].phase else {
            return XCTFail("expected a downloading event")
        }
        XCTAssertEqual(completed, 1)
        XCTAssertEqual(total, 2)
    }

    func testSubdirectoryProgressUsesByteWeightingDuringLargeFile() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 399,
            totalBytes: 665,
            completedFiles: 9,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, 399.0 / 665.0, accuracy: 0.0001)
        XCTAssertGreaterThan(fraction, Double(9) / Double(22))
    }

    func testSubdirectoryProgressIsMonotonicForRealisticSequence() {
        let progress = [
            ProgressReporter.downloadFraction(
                completedBytes: 0,
                totalBytes: 665,
                completedFiles: 0,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 40,
                totalBytes: 665,
                completedFiles: 4,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 120,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 399,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 565,
                totalBytes: 665,
                completedFiles: 10,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 665,
                totalBytes: 665,
                completedFiles: 22,
                totalFiles: 22
            ),
        ]

        XCTAssertEqual(progress, progress.sorted())
    }

    func testSubdirectoryProgressClampsToOne() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 700,
            totalBytes: 665,
            completedFiles: 22,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, 1.0)
    }

    func testSubdirectoryProgressFallsBackToFileCountWhenTotalBytesUnknown() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 9,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, Double(9) / Double(22), accuracy: 0.0001)
    }

    func testSubdirectoryProgressIsCompleteWhenThereAreNoFiles() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 0,
            totalFiles: 0
        )

        XCTAssertEqual(fraction, 1.0)
    }
}
