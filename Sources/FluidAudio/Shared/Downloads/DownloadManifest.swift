import Foundation

struct DownloadFile: Sendable {
    let remotePath: String
    let localRelativePath: String
    let sizeBytes: Int64?

    init(remotePath: String, localRelativePath: String, sizeBytes: Int64?) {
        self.remotePath = remotePath
        self.localRelativePath = localRelativePath
        self.sizeBytes = sizeBytes.flatMap { $0 >= 0 ? $0 : nil }
    }
}

struct DownloadManifest: Sendable {
    let files: [DownloadFile]

    var totalKnownBytes: Int64 {
        files.reduce(0) { $0 + ($1.sizeBytes ?? 0) }
    }

    var hasUnknownSizes: Bool {
        files.contains { $0.sizeBytes == nil }
    }
}

struct ManifestDownloadUnit: Sendable {
    let manifest: DownloadManifest
    let destinationRoot: URL
    let makeRequest: @Sendable (DownloadFile) throws -> URLRequest
}

struct ValidatedDownloadResult: Sendable {
    let finalURL: URL
    let bytesWritten: Int64
    let resumed: Bool
}
