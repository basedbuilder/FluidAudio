@preconcurrency import CoreML
import Foundation

/// Manager for SenseVoiceSmall transcription.
///
/// Pipeline: waveform → [Preprocessor fp32/CPU] → 560-d features → pad to the
/// smallest enumerated encoder bucket → [encoder+CTC fp16/ANE] → greedy CTC
/// decode (drop blank 0, collapse) → SentencePiece detokenize → strip the
/// leading `<|lang|><|emo|><|event|><|itn|>` tags.
public actor SenseVoiceManager {

    private let models: SenseVoiceModels
    private let language: Int32
    private let textNorm: Int32
    private static let logger = AppLogger(category: "SenseVoiceManager")

    public init(
        models: SenseVoiceModels,
        language: Int32 = SenseVoiceConfig.defaultLanguage,
        textNorm: Int32 = SenseVoiceConfig.defaultTextNorm
    ) {
        self.models = models
        self.language = language
        self.textNorm = textNorm
    }

    /// Load models from the default cache (downloading if needed), then build a manager.
    public static func load(
        precision: SenseVoiceEncoderPrecision = .fp16,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> SenseVoiceManager {
        let models = try await SenseVoiceModels.downloadAndLoad(
            precision: precision, progressHandler: progressHandler)
        return SenseVoiceManager(models: models)
    }

    /// Transcribe a 16 kHz mono audio file.
    public func transcribe(audioURL: URL) throws -> String {
        try autoreleasepool {
            let converter = AudioConverter(sampleRate: Double(SenseVoiceConfig.sampleRate))
            let samples = try converter.resampleAudioFile(audioURL)
            return try transcribe(audio: samples)
        }
    }

    /// Transcribe 16 kHz mono float samples (in [-1, 1]).
    public func transcribe(audio: [Float]) throws -> String {
        try autoreleasepool {
            let (features, validFeatureFrames) = try runPreprocessor(audio: audio)
            let (logits, validFrames) = try runEncoder(features: features, validFeatureFrames: validFeatureFrames)
            return decode(logits: logits, validFrames: validFrames)
        }
    }

    // MARK: - Pipeline

    /// waveform [1, N] (scaled to int16 range) → features [1, T, 560].
    private func runPreprocessor(audio: [Float]) throws -> (MLMultiArray, Int) {
        if let nativePreprocessor = models.nativePreprocessor {
            let features = try nativePreprocessor.computeFeatures(audio: audio)
            return (features, features.shape[1].intValue)
        }

        let n = audio.count
        let waveform = try MLMultiArray(shape: [1, n as NSNumber], dataType: .float32)
        let scale = SenseVoiceConfig.waveformScale
        let wptr = waveform.dataPointer.assumingMemoryBound(to: Float32.self)
        let waveformStrides = waveform.strides.map(\.intValue)
        let sampleStride = waveformStrides.count > 1 ? waveformStrides[1] : 1
        for i in 0..<n { wptr[i * sampleStride] = audio[i] * scale }

        let input = try MLDictionaryFeatureProvider(
            dictionary: ["waveform": MLFeatureValue(multiArray: waveform)])
        let out = try models.preprocessor.prediction(from: input)
        guard let features = out.featureValue(for: "features")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice preprocessor produced no `features`")
        }
        return (features, features.shape[1].intValue)
    }

    /// features [1, T, 560] → (ctc_logits [1, bucket+4, V], validFrames = 4 + T).
    private func runEncoder(features: MLMultiArray, validFeatureFrames: Int) throws -> (MLMultiArray, Int) {
        let dim = SenseVoiceConfig.featureDim
        var t = min(features.shape[1].intValue, validFeatureFrames)
        if t > SenseVoiceConfig.maxFrames {
            Self.logger.warning("Audio exceeds max length; truncating \(t) → \(SenseVoiceConfig.maxFrames) frames")
            t = SenseVoiceConfig.maxFrames
        }
        let bucket =
            models.encoderPrecision == .fp32
            ? SenseVoiceConfig.maxFrames
            : SenseVoiceConfig.pickBucket(forFrames: t)

        // Zero-padded [1, bucket, 560] with the first T feature frames copied in.
        let speech = try MLMultiArray(shape: [1, bucket as NSNumber, dim as NSNumber], dataType: .float32)
        let sptr = speech.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(sptr, 0, bucket * dim * MemoryLayout<Float32>.size)
        copyFeatures(features, frames: t, dim: dim, to: sptr)

        let lengths = try MLMultiArray(shape: [1], dataType: .int32)
        lengths[0] = NSNumber(value: t)
        let lang = try MLMultiArray(shape: [1], dataType: .int32)
        lang[0] = NSNumber(value: language)
        let tn = try MLMultiArray(shape: [1], dataType: .int32)
        tn[0] = NSNumber(value: textNorm)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "speech": MLFeatureValue(multiArray: speech),
            "speech_lengths": MLFeatureValue(multiArray: lengths),
            "language": MLFeatureValue(multiArray: lang),
            "textnorm": MLFeatureValue(multiArray: tn),
        ])
        let out = try models.encoder.prediction(from: input)
        guard let logits = out.featureValue(for: "ctc_logits")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice encoder produced no `ctc_logits`")
        }
        return (logits, SenseVoiceConfig.numQueryTokens + t)
    }

    private func copyFeatures(
        _ features: MLMultiArray,
        frames: Int,
        dim: Int,
        to destination: UnsafeMutablePointer<Float32>
    ) {
        let featureStrides = features.strides.map(\.intValue)
        guard featureStrides.count >= 3 else {
            for i in 0..<(frames * dim) { destination[i] = features[i].floatValue }
            return
        }

        let timeStride = featureStrides[1]
        let dimStride = featureStrides[2]

        if features.dataType == .float32 {
            let source = features.dataPointer.assumingMemoryBound(to: Float32.self)
            for frame in 0..<frames {
                let sourceBase = frame * timeStride
                let destBase = frame * dim
                for d in 0..<dim {
                    destination[destBase + d] = source[sourceBase + d * dimStride]
                }
            }
        } else {
            for frame in 0..<frames {
                let destBase = frame * dim
                for d in 0..<dim {
                    destination[destBase + d] = features[[0, frame as NSNumber, d as NSNumber]].floatValue
                }
            }
        }
    }

    /// Greedy CTC over the first `validFrames` (drop blank 0, collapse repeats),
    /// detokenize, then strip the `<|...|>` meta tags.
    private func decode(logits: MLMultiArray, validFrames: Int) -> String {
        let vocab = logits.shape[2].intValue
        let frames = min(validFrames, logits.shape[1].intValue)
        var ids: [Int] = []
        var prev = -1

        func appendArgmax(frameBase: (Int) -> Float) {
            var best = 0
            var bestVal = frameBase(0)
            for v in 1..<vocab {
                let x = frameBase(v)
                if x > bestVal {
                    bestVal = x
                    best = v
                }
            }
            if best != SenseVoiceConfig.blankId && best != prev { ids.append(best) }
            prev = best
        }

        let logitStrides = logits.strides.map(\.intValue)
        let timeStride = logitStrides.count > 1 ? logitStrides[1] : vocab
        let vocabStride = logitStrides.count > 2 ? logitStrides[2] : 1

        if logits.dataType == .float32 {
            let p = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            for t in 0..<frames {
                let base = t * timeStride
                appendArgmax { p[base + $0 * vocabStride] }
            }
        } else if logits.dataType == .float16 {
            let p = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for t in 0..<frames {
                let base = t * timeStride
                appendArgmax { Float(p[base + $0 * vocabStride]) }
            }
        } else {
            for t in 0..<frames {
                appendArgmax { logits[[0, t as NSNumber, $0 as NSNumber]].floatValue }
            }
        }

        let raw = decodeCtcTokenIds(ids, vocabulary: models.vocabulary)
        return
            raw
            .replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }
}
