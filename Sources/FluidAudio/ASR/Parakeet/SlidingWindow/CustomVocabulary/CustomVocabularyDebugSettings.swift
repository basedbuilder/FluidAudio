import Foundation

enum CustomVocabularyDebugSettings {
    static let environmentVariable = "FLUIDAUDIO_CUSTOM_VOCAB_DEBUG"

    static func verboseLoggingEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let rawValue = environment[environmentVariable] else { return false }

        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }
}
