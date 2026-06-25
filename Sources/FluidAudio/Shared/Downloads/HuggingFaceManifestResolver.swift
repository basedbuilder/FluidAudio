import Foundation

struct HuggingFaceManifestResolver: Sendable {
    let session: URLSession

    init(session: URLSession = DownloadUtils.sharedSession) {
        self.session = session
    }

    func resolveRepoManifest(
        repo: Repo,
        variant: String?,
        additionalModelNames: Set<String> = []
    ) async throws -> (manifest: DownloadManifest, requiredModels: Set<String>) {
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
            .union(additionalModelNames)
        let subPath = repo.subPath
        let patterns = requiredModels.map { model in
            if let subPath {
                "\(subPath)/\(model)/"
            } else {
                "\(model)/"
            }
        }

        var filesByPath: [String: DownloadFile] = [:]

        func listDirectory(path: String) async throws {
            let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
            let items = try await fetchTree(repoPath: repo.remotePath, apiPath: apiPath, path: path)

            for item in items {
                guard
                    let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String
                else { continue }

                if itemType == "directory" {
                    let shouldProcess: Bool
                    if let subPath {
                        shouldProcess =
                            itemPath == subPath || itemPath.hasPrefix("\(subPath)/")
                            || patterns.contains { itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/") }
                    } else {
                        shouldProcess =
                            patterns.isEmpty
                            || patterns.contains { itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/") }
                    }

                    if shouldProcess {
                        try await listDirectory(path: itemPath)
                    }
                    continue
                }

                guard itemType == "file" else { continue }

                let shouldInclude: Bool
                if let subPath {
                    let isInSubPath = itemPath.hasPrefix("\(subPath)/")
                    let matchesPattern = patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                    let isMetadata =
                        itemPath.hasSuffix(".json") || itemPath.hasSuffix(".model") || itemPath.hasSuffix(".bin")
                    shouldInclude = isInSubPath && (matchesPattern || isMetadata)
                } else {
                    shouldInclude =
                        patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                        || itemPath.hasSuffix(".json") || itemPath.hasSuffix(".txt")
                }

                if shouldInclude {
                    filesByPath[itemPath] = makeDownloadFile(remotePath: itemPath, subPath: subPath, item: item)
                }
            }
        }

        func listRootFiles(matching names: Set<String>) async throws {
            let items = try await fetchTree(repoPath: repo.remotePath, apiPath: "tree/main", path: "")
            for item in items {
                guard
                    let itemPath = item["path"] as? String,
                    item["type"] as? String == "file",
                    names.contains((itemPath as NSString).lastPathComponent)
                else { continue }

                filesByPath[itemPath] = makeDownloadFile(remotePath: itemPath, subPath: subPath, item: item)
            }
        }

        try await listDirectory(path: subPath ?? "")

        if subPath != nil {
            let collected = Set(filesByPath.keys.map { ($0 as NSString).lastPathComponent })
            let missingAux = requiredModels.filter { model in
                !model.hasSuffix(".mlmodelc") && !model.hasSuffix(".mlpackage")
                    && !collected.contains((model as NSString).lastPathComponent)
            }
            if !missingAux.isEmpty {
                try await listRootFiles(matching: Set(missingAux))
            }
        }

        return (
            DownloadManifest(files: filesByPath.values.sorted { $0.remotePath < $1.remotePath }),
            requiredModels
        )
    }

    func resolveSubdirectoryManifest(
        repo: Repo,
        subdirectory: String,
        shouldSkip: (@Sendable (String) -> Bool)? = nil
    ) async throws -> DownloadManifest {
        var filesByPath: [String: DownloadFile] = [:]

        func listFiles(at path: String) async throws {
            let items = try await fetchTree(repoPath: repo.remotePath, apiPath: "tree/main/\(path)", path: path)
            for item in items {
                guard
                    let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String
                else { continue }

                if shouldSkip?(itemPath) == true {
                    continue
                }

                if itemType == "directory" {
                    try await listFiles(at: itemPath)
                } else if itemType == "file" {
                    filesByPath[itemPath] = makeDownloadFile(remotePath: itemPath, subPath: nil, item: item)
                }
            }
        }

        try await listFiles(at: subdirectory)
        return DownloadManifest(files: filesByPath.values.sorted { $0.remotePath < $1.remotePath })
    }

    private func fetchTree(repoPath: String, apiPath: String, path: String) async throws -> [[String: Any]] {
        let dirURL = try ModelRegistry.apiModels(repoPath, apiPath)
        let request = DownloadUtils.authorizedRequest(url: dirURL)
        let (dirData, response) = try await session.data(for: request)

        if let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 429 || httpResponse.statusCode == 503
        {
            throw DownloadUtils.HuggingFaceDownloadError.rateLimited(
                statusCode: httpResponse.statusCode,
                message: "Rate limited while listing files in \(path)"
            )
        }

        try DownloadUtils.validateJSONResponse(dirData, path: path)

        guard let items = try JSONSerialization.jsonObject(with: dirData) as? [[String: Any]] else {
            throw DownloadUtils.HuggingFaceDownloadError.invalidResponse
        }
        return items
    }

    private func makeDownloadFile(remotePath: String, subPath: String?, item: [String: Any]) -> DownloadFile {
        var localPath = remotePath
        if let subPath, remotePath.hasPrefix("\(subPath)/") {
            localPath = String(remotePath.dropFirst(subPath.count + 1))
        }

        let size = (item["size"] as? NSNumber)?.int64Value ?? -1
        return DownloadFile(remotePath: remotePath, localRelativePath: localPath, sizeBytes: size)
    }
}
