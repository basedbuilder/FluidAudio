import CoreML
import Foundation

/// HuggingFace model downloader using URLSession
public class DownloadUtils {

    private static let logger = AppLogger(category: "DownloadUtils")

    /// Shared URLSession with registry and proxy configuration
    public static let sharedSession: URLSession = ModelRegistry.configuredSession()

    /// Offline-only mode. When true, every public download surface
    /// (`fetchWithAuth`, `downloadRepo`, `downloadSubdirectory`,
    /// `fetchHuggingFaceFile`) and the `loadModels` retry-with-redownload
    /// fallback throws `DownloadUtils.OfflineError` instead of touching
    /// the network. Applications that bundle their own model assets
    /// should set this once at startup and route loading through manual
    /// APIs (e.g. `MLModel(contentsOf:)`, `VadManager(config:vadModel:)`)
    /// so a corrupt-detected `.mlmodelc` never silently re-downloads at
    /// runtime.
    ///
    /// Defaults to `false`. `nonisolated(unsafe)` is acceptable because
    /// the flag is set once at startup before any FluidAudio loaders
    /// are touched and is read-only thereafter.
    nonisolated(unsafe) public static var enforceOffline: Bool = false

    /// Errors thrown when `enforceOffline` is on and FluidAudio would
    /// otherwise attempt a network fetch or a cache rebuild that
    /// requires network. Sibling to `HuggingFaceDownloadError`.
    public enum OfflineError: LocalizedError {
        /// A code path that would have hit the network was blocked.
        /// `operation` is the short tag of the blocked entry point
        /// (e.g. `"downloadRepo(parakeet-tdt-0.6b-v3-coreml)"`).
        case networkDisabled(operation: String)

        /// `loadModels` was invoked but one or more required files are
        /// missing from the local cache. Caller bundled assets but the
        /// bundle was incomplete; surfacing the missing list lets the
        /// caller decide whether to ship a fix or fail loudly.
        case modelMissing(repo: String, missing: [String])

        public var errorDescription: String? {
            switch self {
            case .networkDisabled(let operation):
                return "FluidAudio offline mode: \(operation) blocked"
            case .modelMissing(let repo, let missing):
                return
                    "FluidAudio offline mode: required models missing for \(repo): \(missing.joined(separator: ", "))"
            }
        }
    }

    /// Throws `OfflineError.networkDisabled` if `enforceOffline` is on.
    /// Call this at the top of any path that would touch the network.
    static func ensureOnlineAllowed(_ operation: String) throws {
        if enforceOffline {
            throw OfflineError.networkDisabled(operation: operation)
        }
    }

    /// Get HuggingFace token from environment if available.
    /// Supports multiple env vars for compatibility with different HuggingFace tools:
    /// - HF_TOKEN: Official HuggingFace CLI
    /// - HUGGING_FACE_HUB_TOKEN: Python huggingface_hub library
    /// - HUGGINGFACEHUB_API_TOKEN: LangChain and older integrations
    private static var huggingFaceToken: String? {
        ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGINGFACEHUB_API_TOKEN"]
    }

    /// Create a URLRequest with optional auth header and timeout
    static func authorizedRequest(
        url: URL, timeout: TimeInterval = DownloadConfig.default.timeout
    ) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        if let token = huggingFaceToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// Fetch data from a URL with HuggingFace authentication if available
    /// Use this for API calls that need auth tokens for private repos or higher rate limits
    public static func fetchWithAuth(from url: URL) async throws -> (Data, URLResponse) {
        try ensureOnlineAllowed("fetchWithAuth(\(url.absoluteString))")
        let request = authorizedRequest(url: url)
        return try await sharedSession.data(for: request)
    }

    /// Validate that response data is JSON, not HTML error page
    /// HuggingFace sometimes returns 200 OK with HTML error pages during rate limiting/timeouts
    static func validateJSONResponse(_ data: Data, path: String) throws {
        // Check if response starts with HTML markers
        if let responseString = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) {
            if responseString.hasPrefix("<") || responseString.lowercased().contains("<!doctype html") {
                let snippet = String(responseString.prefix(100))
                throw HuggingFaceDownloadError.htmlErrorResponse(path: path, snippet: snippet)
            }
        }
    }

    public enum HuggingFaceDownloadError: LocalizedError {
        case invalidResponse
        case rateLimited(statusCode: Int, message: String)
        case downloadFailed(path: String, underlying: Error)
        case modelNotFound(path: String)
        case htmlErrorResponse(path: String, snippet: String)

        public var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "Received an invalid response from Hugging Face."
            case .rateLimited(_, let message):
                return "Hugging Face rate limit encountered: \(message)"
            case .downloadFailed(let path, let underlying):
                return "Failed to download \(path): \(underlying.localizedDescription)"
            case .htmlErrorResponse(let path, let snippet):
                return "HuggingFace returned HTML instead of JSON for \(path) (rate limit or server issue): \(snippet)"
            case .modelNotFound(let path):
                return "Model file not found: \(path)"
            }
        }
    }

    /// Phase of a model download operation.
    public enum DownloadPhase: Sendable {
        /// Listing files from the remote repository.
        case listing
        /// Downloading model files. `completedFiles` / `totalFiles` track per-file progress.
        case downloading(completedFiles: Int, totalFiles: Int)
        /// Compiling CoreML models after download.
        case compiling(modelName: String)
    }

    /// Progress snapshot passed to ``ProgressHandler`` closures.
    public struct DownloadProgress: Sendable {
        /// Fraction complete in [0, 1].
        public let fractionCompleted: Double
        /// Current phase of the operation.
        public let phase: DownloadPhase
        /// Bytes downloaded or already present for the current download request.
        public let downloadedBytes: Int64?
        /// Total bytes for the current download request when every manifest entry has a known size.
        public let totalBytes: Int64?
        /// Byte-accurate download fraction in [0, 1] when `totalBytes` is known.
        public let downloadFractionCompleted: Double?
        /// Remote or local relative path of the file currently being downloaded.
        public let currentFile: String?

        public init(
            fractionCompleted: Double,
            phase: DownloadPhase,
            downloadedBytes: Int64? = nil,
            totalBytes: Int64? = nil,
            downloadFractionCompleted: Double? = nil,
            currentFile: String? = nil
        ) {
            self.fractionCompleted = fractionCompleted
            self.phase = phase
            self.downloadedBytes = downloadedBytes
            self.totalBytes = totalBytes
            self.downloadFractionCompleted = downloadFractionCompleted
            self.currentFile = currentFile
        }
    }

    /// Callback type for download progress reporting.
    ///
    /// Called on an unspecified queue. If you need to update UI, dispatch to
    /// the main actor inside your handler.
    public typealias ProgressHandler = @Sendable (DownloadProgress) -> Void

    public struct DownloadConfig: Sendable {
        public let timeout: TimeInterval

        public init(timeout: TimeInterval = 1800) {  // 30 minutes for large models
            self.timeout = timeout
        }

        public static let `default` = DownloadConfig()
    }

    public struct DownloadRequestUnit: Sendable {
        public enum Kind: Sendable {
            case repo(Repo, variant: String?, additionalModelNames: Set<String>)
            case subdirectory(Repo, subdirectory: String)
        }

        public let kind: Kind

        public init(kind: Kind) {
            self.kind = kind
        }

        public static func repo(
            _ repo: Repo,
            variant: String? = nil,
            additionalModelNames: Set<String> = []
        ) -> DownloadRequestUnit {
            DownloadRequestUnit(kind: .repo(repo, variant: variant, additionalModelNames: additionalModelNames))
        }

        public static func subdirectory(_ repo: Repo, subdirectory: String) -> DownloadRequestUnit {
            DownloadRequestUnit(kind: .subdirectory(repo, subdirectory: subdirectory))
        }
    }

    public static func loadModels(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> [String: MLModel] {
        await SystemInfo.logOnce(using: logger)
        do {
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        } catch {
            // In offline mode never delete cache + re-download. Surface
            // the original load failure so the caller can decide.
            if enforceOffline {
                logger.warning(
                    "Offline mode: load failed and re-download blocked. \(error.localizedDescription)"
                )
                throw error
            }

            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)

            // Try to delete the corrupted cache
            do {
                try FileManager.default.removeItem(at: repoPath)
                logger.info("Successfully deleted corrupted cache at \(repoPath.path)")
            } catch {
                // If deletion fails (excluding "file not found"), log the error but continue
                // Robust directory creation will handle any remaining files
                let nsError = error as NSError
                if nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileNoSuchFileError {
                    // File already doesn't exist - this is fine
                } else {
                    logger.warning("Failed to delete cache: \(error.localizedDescription)")
                    logger.info("Will attempt to overwrite during re-download")
                }
            }

            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        }
    }

    public static func clearModelCache(forRepo repo: Repo, directory: URL) {
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try? FileManager.default.removeItem(at: repoPath)
    }

    /// Remove all downloaded models and caches.
    ///
    /// Clears both cache locations:
    /// - `~/Library/Application Support/FluidAudio/Models/` (ASR, VAD, Diarization)
    /// - the shared TTS root: `~/.cache/fluidaudio/` on macOS,
    ///   `Application Support/fluidaudio/` on iOS (matches `TtsCacheDirectory`).
    public static func clearAllModelCaches() {
        let fm = FileManager.default

        // ASR, VAD, Diarization models
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let modelsDir = appSupport.appendingPathComponent("FluidAudio/Models")
            try? fm.removeItem(at: modelsDir)
        }

        // TTS models (Kokoro, PocketTTS, Supertonic3, StyleTTS2).
        // Remove the whole `fluidaudio/` root so every backend subdirectory
        // (Models/, voice packs, etc.) is cleared, not just `Models/`.
        #if os(macOS)
        let home = fm.homeDirectoryForCurrentUser
        let ttsCache = home.appendingPathComponent(".cache/fluidaudio")
        try? fm.removeItem(at: ttsCache)
        #else
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let ttsCache = appSupport.appendingPathComponent("fluidaudio")
            try? fm.removeItem(at: ttsCache)
        }
        #endif

        logger.info("All model caches cleared")
    }

    private static func loadModelsOnce(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> [String: MLModel] {
        await SystemInfo.logOnce(using: logger)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let repoPath = directory.appendingPathComponent(repo.folderName)
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
        // The caller-supplied `modelNames` may include files outside the repo's
        // default "required" set (e.g. CtcHead.mlmodelc inside parakeet-ctc-110m
        // when loaded by the TDT-CTC manager — see issue #524). Union them in
        // so the cache-validity check and the download filter both consider
        // every model the caller actually needs.
        let extraModelNames = Set(modelNames).subtracting(requiredModels)
        let effectiveModels = requiredModels.union(extraModelNames)
        let allModelsExist = effectiveModels.allSatisfy { model in
            let modelPath = repoPath.appendingPathComponent(model)
            return FileManager.default.fileExists(atPath: modelPath.path)
        }

        if !allModelsExist {
            // In offline mode surface a typed error listing the
            // missing files instead of attempting a HuggingFace fetch.
            if enforceOffline {
                let missing = effectiveModels.filter { name in
                    !FileManager.default.fileExists(atPath: repoPath.appendingPathComponent(name).path)
                }.sorted()
                logger.error(
                    "Offline mode: required models missing at \(repoPath.path): \(missing)"
                )
                throw OfflineError.modelMissing(repo: repo.folderName, missing: missing)
            }
            logger.info("Models not found in cache at \(repoPath.path)")
            try await downloadRepo(
                repo, to: directory, variant: variant,
                additionalModelNames: extraModelNames,
                progressHandler: progressHandler)
        } else {
            logger.info("Found \(repo.folderName) locally, no download needed")
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.allowLowPrecisionAccumulationOnGPU = true

        var models: [String: MLModel] = [:]
        for (index, name) in modelNames.enumerated() {
            let modelPath = repoPath.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw CocoaError(
                    .fileNoSuchFile,
                    userInfo: [
                        NSFilePathErrorKey: modelPath.path,
                        NSLocalizedDescriptionKey: "Model file not found: \(name)",
                    ])
            }

            var isDirectory: ObjCBool = false
            guard
                FileManager.default.fileExists(atPath: modelPath.path, isDirectory: &isDirectory),
                isDirectory.boolValue
            else {
                throw CocoaError(
                    .fileReadCorruptFile,
                    userInfo: [
                        NSFilePathErrorKey: modelPath.path,
                        NSLocalizedDescriptionKey: "Model path is not a directory: \(name)",
                    ])
            }

            let coremlDataPath = modelPath.appendingPathComponent("coremldata.bin")
            guard FileManager.default.fileExists(atPath: coremlDataPath.path) else {
                logger.error("Missing coremldata.bin in \(name)")
                throw CocoaError(
                    .fileReadCorruptFile,
                    userInfo: [
                        NSFilePathErrorKey: coremlDataPath.path,
                        NSLocalizedDescriptionKey: "Missing coremldata.bin in model: \(name)",
                    ])
            }

            progressHandler?(
                DownloadProgress(
                    fractionCompleted: 0.5 + 0.5 * Double(index) / Double(modelNames.count),
                    phase: .compiling(modelName: name)
                ))

            let start = Date()
            let model = try MLModel(contentsOf: modelPath, configuration: config)
            let elapsed = Date().timeIntervalSince(start)

            models[name] = model

            let ms = elapsed * 1000
            let formatted = String(format: "%.2f", ms)
            logger.info("Compiled model \(name) in \(formatted) ms :: \(SystemInfo.summary())")
        }

        progressHandler?(DownloadProgress(fractionCompleted: 1.0, phase: .compiling(modelName: "")))
        return models
    }

    /// Download a HuggingFace repository using URLSession (does not load models).
    ///
    /// - Parameter additionalModelNames: Extra model directory names (e.g.
    ///   `"CtcHead.mlmodelc"`) to fetch in addition to the repo's default
    ///   `ModelNames.getRequiredModelNames(...)` set. Used by `loadModels` to
    ///   forward caller-requested files that are not part of the repo's
    ///   baseline required set.
    public static func downloadRepo(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        additionalModelNames: Set<String> = [],
        progressHandler: ProgressHandler? = nil
    ) async throws {
        try ensureOnlineAllowed("downloadRepo(\(repo.folderName))")
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        progressHandler?(DownloadProgress(fractionCompleted: 0.0, phase: .listing))
        let resolver = HuggingFaceManifestResolver()
        let resolved = try await resolver.resolveRepoManifest(
            repo: repo,
            variant: variant,
            additionalModelNames: additionalModelNames
        )
        logger.info("Found \(resolved.manifest.files.count) files to download")

        let unit = ManifestDownloadUnit(
            manifest: resolved.manifest,
            destinationRoot: repoPath,
            makeRequest: { file in
                try Self.downloadRequest(repo: repo, remotePath: file.remotePath)
            }
        )
        try await ManifestDownloadRunner().run(
            units: [unit],
            legacyFractionMultiplier: 0.5,
            progress: { progressHandler?($0) }
        )

        // Verify required models are present
        for model in resolved.requiredModels {
            let modelPath = repoPath.appendingPathComponent(model)
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw HuggingFaceDownloadError.modelNotFound(path: model)
            }
        }

        logger.info("Downloaded all required models for \(repo.folderName)")
    }

    public static func download(
        _ units: [DownloadRequestUnit],
        to directory: URL,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        try ensureOnlineAllowed("download(\(units.count) units)")
        progressHandler?(DownloadProgress(fractionCompleted: 0.0, phase: .listing))

        let resolver = HuggingFaceManifestResolver()
        var runnerUnits: [ManifestDownloadUnit] = []
        var repoVerifications: [(repoPath: URL, requiredModels: Set<String>)] = []

        for unit in units {
            switch unit.kind {
            case .repo(let repo, let variant, let additionalModelNames):
                let repoPath = directory.appendingPathComponent(repo.folderName)
                try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
                let resolved = try await resolver.resolveRepoManifest(
                    repo: repo,
                    variant: variant,
                    additionalModelNames: additionalModelNames
                )
                runnerUnits.append(
                    ManifestDownloadUnit(
                        manifest: resolved.manifest,
                        destinationRoot: repoPath,
                        makeRequest: { file in
                            try Self.downloadRequest(repo: repo, remotePath: file.remotePath)
                        }
                    ))
                repoVerifications.append((repoPath, resolved.requiredModels))

            case .subdirectory(let repo, let subdirectory):
                let repoPath = directory.appendingPathComponent(repo.folderName)
                try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
                let manifest = try await resolver.resolveSubdirectoryManifest(
                    repo: repo,
                    subdirectory: subdirectory
                )
                runnerUnits.append(
                    ManifestDownloadUnit(
                        manifest: manifest,
                        destinationRoot: repoPath,
                        makeRequest: { file in
                            try Self.downloadRequest(repo: repo, remotePath: file.remotePath)
                        }
                    ))
            }
        }

        try await ManifestDownloadRunner().run(
            units: runnerUnits,
            legacyFractionMultiplier: 1.0,
            progress: { progressHandler?($0) }
        )

        for verification in repoVerifications {
            for model in verification.requiredModels {
                let modelPath = verification.repoPath.appendingPathComponent(model)
                guard FileManager.default.fileExists(atPath: modelPath.path) else {
                    throw HuggingFaceDownloadError.modelNotFound(path: model)
                }
            }
        }
    }

    // MARK: - Helper Functions

    /// Robustly create a directory, removing any conflicting files in the path.
    ///
    /// This handles cases where a file exists where a directory should be, which can happen
    /// during corrupted cache recovery when partial deletion leaves files in place of directories.
    ///
    /// - Parameter url: The directory path to create
    /// - Throws: Errors from FileManager if directory creation fails after cleanup
    static func createDirectoryRobustly(at url: URL) throws {
        let fm = FileManager.default
        var pathComponents = url.pathComponents

        // Remove leading "/" if present
        if pathComponents.first == "/" {
            pathComponents.removeFirst()
        }

        // Build path incrementally, checking each component
        var currentPath = "/"
        for component in pathComponents {
            currentPath = (currentPath as NSString).appendingPathComponent(component)
            let componentURL = URL(fileURLWithPath: currentPath)

            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: currentPath, isDirectory: &isDirectory) {
                if !isDirectory.boolValue {
                    // A file exists where a directory should be - remove it
                    logger.warning("Removing file blocking directory creation: \(currentPath)")
                    try fm.removeItem(at: componentURL)
                    try fm.createDirectory(at: componentURL, withIntermediateDirectories: false)
                }
                // If it's already a directory, continue
            } else {
                // Path doesn't exist, create remaining path with intermediate directories
                try fm.createDirectory(at: url, withIntermediateDirectories: true)
                return
            }
        }
    }

    static func downloadRequest(repo: Repo, remotePath: String) throws -> URLRequest {
        let encodedPath = remotePath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? remotePath
        let fileURL = try ModelRegistry.resolveModel(repo.remotePath, encodedPath)
        return authorizedRequest(url: fileURL)
    }

    /// Download a specific subdirectory from a HuggingFace repository.
    ///
    /// Use this for optional model components that aren't part of the required model set
    /// (e.g., the Mimi encoder for PocketTTS voice cloning).
    ///
    /// - Parameters:
    ///   - repo: The HuggingFace repository.
    ///   - subdirectory: Path within the repo to download (e.g. `"mimi_encoder.mlmodelc"`).
    ///   - repoDirectory: Local directory corresponding to the repo root.
    ///     Files are saved at `repoDirectory/<remote_path>`.
    ///   - shouldSkip: Optional predicate evaluated on each remote path
    ///     (both files and directories). Returning `true` excludes the file
    ///     or, for directories, skips the whole subtree without recursing.
    ///     Used to avoid pulling redundant artifacts (e.g. `.mlpackage`
    ///     sources next to compiled `.mlmodelc`).
    public static func downloadSubdirectory(
        _ repo: Repo,
        subdirectory: String,
        to repoDirectory: URL,
        progressHandler: ProgressHandler? = nil,
        shouldSkip: (@Sendable (String) -> Bool)? = nil
    ) async throws {
        try ensureOnlineAllowed("downloadSubdirectory(\(repo.folderName)/\(subdirectory))")
        progressHandler?(DownloadProgress(fractionCompleted: 0.0, phase: .listing))
        let resolver = HuggingFaceManifestResolver()
        let manifest = try await resolver.resolveSubdirectoryManifest(
            repo: repo,
            subdirectory: subdirectory,
            shouldSkip: shouldSkip
        )
        logger.info("Found \(manifest.files.count) files in \(subdirectory)")

        let unit = ManifestDownloadUnit(
            manifest: manifest,
            destinationRoot: repoDirectory,
            makeRequest: { file in
                try Self.downloadRequest(repo: repo, remotePath: file.remotePath)
            }
        )
        try await ManifestDownloadRunner().run(
            units: [unit],
            legacyFractionMultiplier: 1.0,
            progress: { progressHandler?($0) }
        )
        logger.info("Downloaded \(subdirectory) from \(repo.folderName)")
    }

    /// Fetch a single file from HuggingFace with retry
    public static func fetchHuggingFaceFile(
        from url: URL,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0
    ) async throws -> Data {
        try ensureOnlineAllowed("fetchHuggingFaceFile(\(description))")
        var lastError: Error?
        let request = authorizedRequest(url: url)

        for attempt in 1...maxAttempts {
            do {
                let (data, response) = try await sharedSession.data(for: request)

                guard let httpResponse = response as? HTTPURLResponse else {
                    throw HuggingFaceDownloadError.invalidResponse
                }

                if httpResponse.statusCode == 429 || httpResponse.statusCode == 503 {
                    throw HuggingFaceDownloadError.rateLimited(
                        statusCode: httpResponse.statusCode,
                        message: "HTTP \(httpResponse.statusCode)"
                    )
                }

                guard (200..<300).contains(httpResponse.statusCode) else {
                    throw HuggingFaceDownloadError.invalidResponse
                }

                return data

            } catch {
                lastError = error
                if attempt < maxAttempts {
                    let backoffSeconds = pow(2.0, Double(attempt - 1)) * minBackoff
                    logger.warning(
                        "Download attempt \(attempt) for \(description) failed: \(error.localizedDescription). Retrying in \(String(format: "%.1f", backoffSeconds))s."
                    )
                    try await Task.sleep(nanoseconds: UInt64(backoffSeconds * 1_000_000_000))
                }
            }
        }

        throw lastError ?? HuggingFaceDownloadError.invalidResponse
    }
}
