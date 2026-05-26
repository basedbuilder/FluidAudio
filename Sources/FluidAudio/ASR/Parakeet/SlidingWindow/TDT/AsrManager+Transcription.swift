import CoreML
import Foundation

extension AsrManager {

    internal func transcribeWithState(
        _ audioSamples: [Float],
        decoderState: inout TdtDecoderState,
        language: Language? = nil,
        customVocabulary: CustomVocabularyContext? = nil,
        vocabularyRescorer: VocabularyRescorer? = nil
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard audioSamples.count >= minimumRequiredSamples else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Route to appropriate processing method based on audio length
        if audioSamples.count <= ASRConstants.maxModelSamples {
            let (alignedSamples, frameAlignedLength) = frameAlignedAudio(audioSamples)
            let paddedAudio: [Float] = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
            let shouldComputeCtcHead = customVocabulary != nil && vocabularyRescorer != nil
            let (hypothesis, encoderSequenceLength, ctcHeadEncoderOutputRetained) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState,
                isLastChunk: true,  // Single-chunk: always first and last
                language: language,
                includeCtcHeadLogProbs: shouldComputeCtcHead
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

            return try await applyCtcHeadRescoringIfNeeded(
                to: result,
                ctcHeadEncoderOutputRetained: ctcHeadEncoderOutputRetained,
                audioSampleCount: frameAlignedLength,
                customVocabulary: customVocabulary,
                vocabularyRescorer: vocabularyRescorer
            )
        }

        if customVocabulary != nil || vocabularyRescorer != nil {
            throw ASRError.processingFailed("CTC-head custom vocabulary rescoring is only available for single-window audio")
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
        let (hypothesis, encLen, _) = try await executeMLInferenceWithTimings(
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

    private func applyCtcHeadRescoringIfNeeded(
        to result: ASRResult,
        ctcHeadEncoderOutputRetained: Bool,
        audioSampleCount: Int,
        customVocabulary: CustomVocabularyContext?,
        vocabularyRescorer: VocabularyRescorer?
    ) async throws -> ASRResult {
        guard let customVocabulary, let vocabularyRescorer else {
            return result
        }
        guard let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty else {
            return result
        }

        let vocabConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: customVocabulary.terms.count)
        let minSimilarity = max(vocabConfig.minSimilarity, customVocabulary.minSimilarity)
        guard vocabularyRescorer.hasPotentialCtcTokenRescoreCandidate(
            transcript: result.text,
            tokenTimings: tokenTimings,
            minSimilarity: minSimilarity
        ) else {
            return result
        }

        defer { retainedCtcHeadEncoderOutput = nil }
        guard ctcHeadEncoderOutputRetained, let ctcHeadEncoderOutput = retainedCtcHeadEncoderOutput else {
            throw ASRError.processingFailed("CTC-head custom vocabulary requested but no encoder output was retained")
        }
        let ctcHeadLogProbs = try await computeCtcHeadLogProbs(
            encoderOutput: ctcHeadEncoderOutput,
            audioSampleCount: audioSampleCount
        )
        guard !ctcHeadLogProbs.logProbs.isEmpty else {
            throw ASRError.processingFailed("CTC-head custom vocabulary requested but no CTC log-probs were produced")
        }

        let rescored = vocabularyRescorer.ctcTokenRescore(
            transcript: result.text,
            tokenTimings: tokenTimings,
            logProbs: ctcHeadLogProbs.logProbs,
            frameDuration: ctcHeadLogProbs.frameDuration,
            cbw: vocabConfig.cbw,
            marginSeconds: ContextBiasingConstants.defaultMarginSeconds,
            minSimilarity: minSimilarity
        )

        let detectedTerms = uniquePreservingOrder(
            rescored.replacements.compactMap(\.replacementWord)
        )
        let appliedTerms = uniquePreservingOrder(
            rescored.replacements
                .filter(\.shouldReplace)
                .compactMap(\.replacementWord)
        )

        return result.withRescoring(
            text: rescored.text,
            detected: detectedTerms.isEmpty ? nil : detectedTerms,
            applied: appliedTerms.isEmpty ? nil : appliedTerms
        )
    }

    private func uniquePreservingOrder(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var deduped: [String] = []
        deduped.reserveCapacity(values.count)

        for value in values where seen.insert(value).inserted {
            deduped.append(value)
        }

        return deduped
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

}
