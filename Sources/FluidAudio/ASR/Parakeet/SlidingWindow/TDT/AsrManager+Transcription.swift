import Foundation

extension AsrManager {

    /// Transcribe one model window and rescore plausible vocabulary spans with the fused CTC head.
    public func transcribe(
        _ audioSamples: [Float],
        decoderState: inout TdtDecoderState,
        language: Language? = nil,
        customVocabulary: CustomVocabularyContext,
        vocabularyRescorer: VocabularyRescorer
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard audioSamples.count >= minimumRequiredSamples else { throw ASRError.invalidAudioData }
        guard audioSamples.count <= ASRConstants.maxModelSamples else {
            throw ASRError.processingFailed(
                "CTC-head custom vocabulary rescoring is only available for single-window audio"
            )
        }

        let startTime = Date()
        let (alignedSamples, frameAlignedLength) = frameAlignedAudio(audioSamples)
        let paddedAudio = padAudioIfNeeded(
            alignedSamples,
            targetLength: ASRConstants.maxModelSamples
        )
        let (hypothesis, encoderSequenceLength, encoderOutput) =
            try await executeMLInferenceWithLocalEncoderOutput(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,
                decoderState: &decoderState,
                isLastChunk: true,
                language: language
            )

        let result = processTranscriptionResult(
            tokenIds: hypothesis.ySequence,
            timestamps: hypothesis.timestamps,
            confidences: hypothesis.tokenConfidences,
            tokenDurations: hypothesis.tokenDurations,
            encoderSequenceLength: encoderSequenceLength,
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
        guard let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty else {
            return result
        }

        let rescorerConfig = ContextBiasingConstants.rescorerConfig(
            forVocabSize: customVocabulary.terms.count
        )
        let minSimilarity = max(rescorerConfig.minSimilarity, customVocabulary.minSimilarity)
        guard vocabularyRescorer.hasPotentialCtcTokenRescoreCandidate(
            transcript: result.text,
            tokenTimings: tokenTimings,
            minSimilarity: minSimilarity
        ) else {
            return result
        }
        guard let encoderOutput else {
            throw ASRError.processingFailed(
                "CTC-head custom vocabulary requested but no encoder output was produced"
            )
        }

        let ctc = try await computeCtcHeadLogProbs(
            encoderOutput: encoderOutput,
            audioSampleCount: frameAlignedLength
        )
        guard !ctc.logProbs.isEmpty else {
            throw ASRError.processingFailed(
                "CTC-head custom vocabulary requested but no CTC log-probabilities were produced"
            )
        }

        let rescored = vocabularyRescorer.ctcTokenRescore(
            transcript: result.text,
            tokenTimings: tokenTimings,
            logProbs: ctc.logProbs,
            frameDuration: ctc.frameDuration,
            cbw: rescorerConfig.cbw,
            marginSeconds: ContextBiasingConstants.defaultMarginSeconds,
            minSimilarity: minSimilarity
        )
        let detected = uniquePreservingOrder(
            rescored.replacements.compactMap(\.replacementWord)
        )
        let applied = uniquePreservingOrder(
            rescored.replacements.filter(\.shouldReplace).compactMap(\.replacementWord)
        )

        return result.withRescoring(
            text: rescored.text,
            detected: detected.isEmpty ? nil : detected,
            applied: applied.isEmpty ? nil : applied
        )
    }

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout TdtDecoderState, language: Language? = nil
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard audioSamples.count >= minimumRequiredSamples else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Route to appropriate processing method based on audio length
        if audioSamples.count <= ASRConstants.maxModelSamples {
            let (alignedSamples, frameAlignedLength) = frameAlignedAudio(audioSamples)
            let paddedAudio: [Float] = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
            let (hypothesis, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState,
                isLastChunk: true,  // Single-chunk: always first and last
                language: language
            )

            let result = processTranscriptionResult(
                tokenIds: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                tokenDurations: hypothesis.tokenDurations,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )

            return result
        }

        // ChunkProcessor handles stateless chunked transcription for long audio
        let processor = ChunkProcessor(audioSamples: audioSamples)
        let result = try await processor.process(
            using: self,
            startTime: startTime,
            progressHandler: { [weak self] progress in
                guard let self else { return }
                await self.progressEmitter.report(progress: progress)
            },
            language: language
        )

        return result
    }

    /// Chunk transcription that preserves decoder state between calls.
    /// Used by SlidingWindowAsrManager for overlapping-window processing with token deduplication.
    func transcribeChunk(
        _ chunkSamples: [Float],
        decoderState: inout TdtDecoderState,
        previousTokens: [Int] = [],
        isLastChunk: Bool = false,
        language: Language? = nil
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], encoderSequenceLength: Int) {
        let (alignedSamples, frameAlignedLength) = frameAlignedAudio(
            chunkSamples, allowAlignment: previousTokens.isEmpty)
        let padded = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        let (hypothesis, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: frameAlignedLength,
            actualAudioFrames: nil,  // Will be calculated from originalLength
            decoderState: &decoderState,
            contextFrameAdjustment: 0,  // Non-streaming chunks don't use adaptive context
            isLastChunk: isLastChunk,
            language: language
        )

        // Apply token deduplication if previous tokens are provided
        if !previousTokens.isEmpty && hypothesis.hasTokens {
            let (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: previousTokens, current: hypothesis.ySequence)
            let adjustedTimestamps =
                removedCount > 0 ? Array(hypothesis.timestamps.dropFirst(removedCount)) : hypothesis.timestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(hypothesis.tokenConfidences.dropFirst(removedCount)) : hypothesis.tokenConfidences

            return (deduped, adjustedTimestamps, adjustedConfidences, encLen)
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, encLen)
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        confidences: [Float] = [],
        tokenDurations: [Int] = [],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval
    ) -> ASRResult {

        let text = convertTokensToText(tokenIds)
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)

        let resultTimings = createTokenTimings(
            from: tokenIds, timestamps: timestamps, confidences: confidences, tokenDurations: tokenDurations)

        let confidence = calculateConfidence(
            tokenCount: tokenIds.count,
            isEmpty: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            tokenConfidences: confidences
        )

        return ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: resultTimings
        )
    }

    private func uniquePreservingOrder(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { seen.insert($0).inserted }
    }

}
