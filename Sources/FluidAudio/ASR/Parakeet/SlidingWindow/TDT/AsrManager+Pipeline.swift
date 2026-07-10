@preconcurrency import CoreML
import Foundation

extension AsrManager {

    internal func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        actualAudioFrames: Int? = nil,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0,
        language: Language? = nil,
        emitTokensAfterGlobalFrame: Int? = nil,
        initialTimeIndexOverride: Int? = nil
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {
        let result = try await executeMLInference(
            paddedAudio,
            originalLength: originalLength,
            actualAudioFrames: actualAudioFrames,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset,
            language: language,
            emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
            initialTimeIndexOverride: initialTimeIndexOverride,
            includeEncoderOutput: false
        )
        return (result.hypothesis, result.encoderSequenceLength)
    }

    internal func executeMLInferenceWithLocalEncoderOutput(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        actualAudioFrames: Int? = nil,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0,
        language: Language? = nil,
        emitTokensAfterGlobalFrame: Int? = nil,
        initialTimeIndexOverride: Int? = nil
    ) async throws -> (
        hypothesis: TdtHypothesis,
        encoderSequenceLength: Int,
        encoderOutput: MLMultiArray?
    ) {
        try await executeMLInference(
            paddedAudio,
            originalLength: originalLength,
            actualAudioFrames: actualAudioFrames,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset,
            language: language,
            emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
            initialTimeIndexOverride: initialTimeIndexOverride,
            includeEncoderOutput: true
        )
    }

    private func executeMLInference(
        _ paddedAudio: [Float],
        originalLength: Int?,
        actualAudioFrames: Int?,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int,
        isLastChunk: Bool,
        globalFrameOffset: Int,
        language: Language?,
        emitTokensAfterGlobalFrame: Int?,
        initialTimeIndexOverride: Int?,
        includeEncoderOutput: Bool
    ) async throws -> (
        hypothesis: TdtHypothesis,
        encoderSequenceLength: Int,
        encoderOutput: MLMultiArray?
    ) {

        let preprocessorInput = try await preparePreprocessorInput(
            paddedAudio, actualLength: originalLength)

        let preprocessorAudioArray = preprocessorInput.featureValue(for: "audio_signal")?.multiArrayValue

        do {
            guard let preprocessorModel = preprocessorModel else {
                throw ASRError.notInitialized
            }

            try Task.checkCancellation()
            let preprocessorOutput = try await preprocessorModel.compatPrediction(
                from: preprocessorInput,
                options: predictionOptions
            )

            let encoderOutputProvider: MLFeatureProvider
            if let encoderModel = encoderModel {
                // Split frontend: run separate encoder
                let encoderInput = try prepareEncoderInput(
                    encoder: encoderModel,
                    preprocessorOutput: preprocessorOutput,
                    originalInput: preprocessorInput
                )

                try Task.checkCancellation()
                encoderOutputProvider = try await encoderModel.compatPrediction(
                    from: encoderInput,
                    options: predictionOptions
                )
            } else {
                // Fused frontend: preprocessor output already contains encoder features
                encoderOutputProvider = preprocessorOutput
            }

            let rawEncoderOutput = try extractFeatureValue(
                from: encoderOutputProvider, key: "encoder", errorMessage: "Invalid encoder output")
            let encoderLength = try extractFeatureValue(
                from: encoderOutputProvider, key: "encoder_length",
                errorMessage: "Invalid encoder output length")

            let encoderSequenceLength = encoderLength[0].intValue

            // Calculate actual audio frames if not provided using shared constants
            let actualFrames =
                actualAudioFrames ?? ASRConstants.calculateEncoderFrames(from: originalLength ?? paddedAudio.count)

            let hypothesis = try await tdtDecodeWithTimings(
                encoderOutput: rawEncoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualFrames,
                originalAudioSamples: paddedAudio,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset,
                language: language,
                emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
                initialTimeIndexOverride: initialTimeIndexOverride
            )

            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }

            return (
                hypothesis,
                encoderSequenceLength,
                includeEncoderOutput ? rawEncoderOutput : nil
            )
        } catch {
            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            throw error
        }
    }

    internal func computeCtcHeadLogProbs(
        encoderOutput: MLMultiArray,
        audioSampleCount: Int
    ) async throws -> (logProbs: [[Float]], frameDuration: Double) {
        guard let models = asrModels else {
            throw ASRError.notInitialized
        }
        guard models.version == .tdtCtc110m else {
            throw ASRError.processingFailed("CTC-head rescoring requires TDT-CTC-110M")
        }
        guard let ctcHead = models.ctcHead else {
            throw ASRError.processingFailed("CTC-head model unavailable for TDT-CTC-110M")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_output": MLFeatureValue(multiArray: encoderOutput)
        ])
        let output = try await ctcHead.compatPrediction(
            from: input,
            options: predictionOptions
        )
        guard let logits = output.featureValue(for: "ctc_logits")?.multiArrayValue else {
            let available = output.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Missing ctc_logits from CTC head output. Available outputs: \(available)"
            )
        }

        let allLogProbs = try ctcHeadLogProbs(from: logits, blankId: models.version.blankId)
        guard !allLogProbs.isEmpty else { return ([], 0) }

        let sampleCount = min(max(audioSampleCount, 0), ASRConstants.maxModelSamples)
        let samplesPerFrame = Double(ASRConstants.maxModelSamples) / Double(allLogProbs.count)
        let validFrames = Int(ceil(Double(sampleCount) / samplesPerFrame))
        let frameCount = max(1, min(validFrames, allLogProbs.count))
        let trimmed = Array(allLogProbs.prefix(frameCount))
        let frameDuration =
            Double(sampleCount) / Double(frameCount) / Double(ASRConstants.sampleRate)
        return (trimmed, frameDuration)
    }

    private func ctcHeadLogProbs(
        from output: MLMultiArray,
        blankId: Int
    ) throws -> [[Float]] {
        let rank = output.shape.count
        guard rank == 3 || rank == 4 else {
            throw ASRError.processingFailed("Unexpected CTC output rank: \(output.shape)")
        }

        let timeSteps: Int
        let vocabularySize: Int
        let index: (Int, Int) -> [NSNumber]
        if rank == 3 {
            timeSteps = output.shape[1].intValue
            vocabularySize = output.shape[2].intValue
            index = { time, token in [0, time, token].map { NSNumber(value: $0) } }
        } else {
            vocabularySize = output.shape[1].intValue
            timeSteps = output.shape[3].intValue
            index = { time, token in [0, token, 0, time].map { NSNumber(value: $0) } }
        }
        guard timeSteps > 0, vocabularySize > 0 else { return [] }

        let temperature = ContextBiasingConstants.ctcTemperature
        let blankBias = ContextBiasingConstants.blankBias
        var result = [[Float]]()
        result.reserveCapacity(timeSteps)

        for time in 0..<timeSteps {
            var logits = [Float]()
            logits.reserveCapacity(vocabularySize)
            for token in 0..<vocabularySize {
                logits.append(output[index(time, token)].floatValue / temperature)
            }

            let maximum = logits.max() ?? 0
            let logSum = logf(logits.reduce(0) { $0 + expf($1 - maximum) })
            var row = logits.map { ($0 - maximum) - logSum }
            if blankBias != 0, row.indices.contains(blankId) {
                row[blankId] -= blankBias
            }
            result.append(row)
        }

        return result
    }

    private func prepareEncoderInput(
        encoder: MLModel,
        preprocessorOutput: MLFeatureProvider,
        originalInput: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        let inputDescriptions = encoder.modelDescription.inputDescriptionsByName

        let missingNames = inputDescriptions.keys.filter { name in
            preprocessorOutput.featureValue(for: name) == nil
        }

        if missingNames.isEmpty {
            return preprocessorOutput
        }

        var features: [String: MLFeatureValue] = [:]

        for name in inputDescriptions.keys {
            if let value = preprocessorOutput.featureValue(for: name) {
                features[name] = value
                continue
            }

            if let fallback = originalInput.featureValue(for: name) {
                features[name] = fallback
                continue
            }

            let availableInputs = preprocessorOutput.featureNames.sorted().joined(separator: ", ")
            let fallbackInputs = originalInput.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Missing required encoder input: \(name). Available inputs: \(availableInputs), "
                    + "fallback inputs: \(fallbackInputs)"
            )
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Align audio samples to encoder frame boundaries by zero-padding to the next frame boundary.
    /// Returns the aligned samples and the frame-aligned length.
    /// - Parameters:
    ///   - audioSamples: Raw audio samples
    ///   - allowAlignment: When false, skip alignment (e.g. when previous context exists)
    nonisolated internal func frameAlignedAudio(
        _ audioSamples: [Float], allowAlignment: Bool = true
    ) -> (samples: [Float], frameAlignedLength: Int) {
        let originalLength = audioSamples.count
        let frameAlignedCandidate =
            ((originalLength + ASRConstants.samplesPerEncoderFrame - 1)
                / ASRConstants.samplesPerEncoderFrame) * ASRConstants.samplesPerEncoderFrame
        if allowAlignment && frameAlignedCandidate > originalLength
            && frameAlignedCandidate <= ASRConstants.maxModelSamples
        {
            let aligned = audioSamples + Array(repeating: 0, count: frameAlignedCandidate - originalLength)
            return (aligned, frameAlignedCandidate)
        }
        return (audioSamples, originalLength)
    }

    nonisolated internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }

}
