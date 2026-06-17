#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `sensevoice-transcribe <audio>... [--language en] [--fp32] [--verbose]`
enum SenseVoiceTranscribeCommand {
    private static let logger = AppLogger(category: "SenseVoiceTranscribe")

    static func run(arguments: [String]) async {
        var audioPaths: [String] = []
        var precision: SenseVoiceEncoderPrecision = .fp16
        var language = SenseVoiceConfig.defaultLanguage
        var verbose = false

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--int8":
                precision = .int8
            case "--fp32":
                precision = .fp32
            case "--language":
                if i + 1 < arguments.count {
                    guard let embedding = SenseVoiceConfig.languageEmbedding(for: arguments[i + 1]) else {
                        logger.error("Unsupported SenseVoice language: \(arguments[i + 1])")
                        return
                    }
                    language = embedding
                    i += 1
                }
            case "--verbose", "-v":
                verbose = true
            case "--help", "-h":
                printUsage()
                return
            default:
                audioPaths.append(arguments[i])
            }
            i += 1
        }

        guard !audioPaths.isEmpty else {
            logger.error("Error: No audio file specified")
            printUsage()
            return
        }
        let audioURLs = audioPaths.map { URL(fileURLWithPath: $0) }
        for audioURL in audioURLs {
            guard FileManager.default.fileExists(atPath: audioURL.path) else {
                logger.error("Error: Audio file not found: \(audioURL.path)")
                return
            }
        }

        do {
            logger.info("Loading SenseVoice models (encoder: \(precision.rawValue))...")
            let models = try await SenseVoiceModels.downloadAndLoad(precision: precision)
            let manager = SenseVoiceManager(models: models, language: language)

            for audioURL in audioURLs {
                let start = Date()
                let text = try await manager.transcribe(audioURL: audioURL)
                let elapsed = Date().timeIntervalSince(start)

                if verbose { logger.info("Transcribed \(audioURL.lastPathComponent) in \(String(format: "%.2f", elapsed))s") }
                if audioURLs.count > 1 {
                    print("[\(audioURL.lastPathComponent)] \(text)")
                } else {
                    print(text)
                }
            }
        } catch {
            logger.error("Transcription failed: \(error)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio sensevoice-transcribe <audio-file>... [options]

            Transcribe audio with SenseVoiceSmall (multilingual, non-autoregressive).

            Options:
              --int8        Use the int8 encoder (~half size, ANE, accuracy-neutral)
              --fp32        Use the fp32 encoder (for hardware without a Neural Engine)
              --language L  Force language: auto, en, zh, yue, ja, ko
              --verbose,-v  Print timing
              --help,-h     Show this help
            """
        )
    }
}
#endif
