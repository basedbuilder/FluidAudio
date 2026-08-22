#if canImport(CNemoTextProcessing)
import CNemoTextProcessing
#else
import Darwin
#endif
import Foundation
import NaturalLanguage

/// Inverse Text Normalization (ITN) for post-processing ASR output.
///
/// Converts spoken-form text to written form:
/// - "two hundred thirty two" → "232"
/// - "five dollars and fifty cents" → "$5.50"
/// - "january fifth twenty twenty five" → "January 5, 2025"
/// - "period" → "."
///
/// Supports three modes:
/// - `normalize(_:)` — single expression normalization
/// - `normalizeSentence(_:)` — sentence-mode with sliding window span matching
/// - `normalizeSentence(_:maxSpanTokens:)` — sentence-mode with custom span size
///
/// Uses Apple NaturalLanguage framework to avoid false positives on ambiguous words
/// (e.g., "period" as a noun vs. punctuation).
///
/// SwiftPM links the native engine (`text-processing-rs`) through the bundled
/// binary target. Other integrations, including CocoaPods, retain the legacy
/// runtime-symbol lookup and degrade to pass-through behavior when it is absent.
public final class TextNormalizer: Sendable {

    /// Whether the native NeMo library is available.
    public let isNativeAvailable: Bool

    /// Whether the linked library exposes the TN (written→spoken) surface used
    /// by the TTS frontends.
    public var isTnAvailable: Bool {
        isNativeAvailable && nemoTnNormalizeSentence != nil
    }

    /// Shared instance for convenience.
    public static let shared = TextNormalizer()

    /// Words that are ambiguous — they could be punctuation spoken forms OR normal English words.
    /// When these appear in sentence context, NLTagger is used to check if they're nouns/verbs/adjectives
    /// (natural language) vs. standalone punctuation commands.
    private static let ambiguousWords: Set<String> = [
        "period", "dash", "colon", "pipe", "slash", "dot", "plus", "hash", "percent",
    ]

    /// Ambiguous words that are operators when they appear between numeric operands.
    private static let numericInfixWords: Set<String> = ["dot", "plus"]

    private typealias StringTransform =
        @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
    private typealias SentenceOptionsTransform =
        @convention(c) (UnsafePointer<CChar>?, UInt32, UInt32, UInt32) -> UnsafeMutablePointer<CChar>?
    private typealias SentenceMaxSpanTransform =
        @convention(c) (UnsafePointer<CChar>?, UInt32) -> UnsafeMutablePointer<CChar>?
    private typealias FreeString = @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
    private typealias AddRule =
        @convention(c) (UnsafePointer<CChar>?, UnsafePointer<CChar>?) -> Void
    private typealias RemoveRule = @convention(c) (UnsafePointer<CChar>?) -> Int32
    private typealias ClearRules = @convention(c) () -> Void
    private typealias RuleCount = @convention(c) () -> UInt32
    private typealias Version = @convention(c) () -> UnsafePointer<CChar>?

    private let nemoNormalize: StringTransform?
    private let nemoNormalizeSentence: StringTransform?
    private let nemoNormalizeSentenceWithOptions: SentenceOptionsTransform?
    private let nemoNormalizeSentenceWithMaxSpan: SentenceMaxSpanTransform?
    private let nemoTnNormalize: StringTransform?
    private let nemoTnNormalizeSentence: StringTransform?
    private let nemoFreeString: FreeString?
    private let nemoAddRule: AddRule?
    private let nemoRemoveRule: RemoveRule?
    private let nemoClearRules: ClearRules?
    private let nemoRuleCount: RuleCount?
    private let nemoVersion: Version?

    public init() {
        #if canImport(CNemoTextProcessing)
        self.nemoNormalize = { nemo_normalize($0) }
        self.nemoNormalizeSentence = { nemo_normalize_sentence($0) }
        self.nemoNormalizeSentenceWithOptions = {
            nemo_normalize_sentence_with_options($0, $1, $2, $3)
        }
        self.nemoNormalizeSentenceWithMaxSpan = nil
        self.nemoTnNormalize = { nemo_tn_normalize($0) }
        self.nemoTnNormalizeSentence = { nemo_tn_normalize_sentence($0) }
        self.nemoFreeString = { nemo_free_string($0) }
        self.nemoAddRule = { nemo_add_rule($0, $1) }
        self.nemoRemoveRule = { nemo_remove_rule($0) }
        self.nemoClearRules = { nemo_clear_rules() }
        self.nemoRuleCount = { nemo_rule_count() }
        self.nemoVersion = { nemo_version() }
        self.isNativeAvailable = true
        #else
        let handle = dlopen(nil, RTLD_NOW)
        self.nemoNormalize = Self.resolve("nemo_normalize", from: handle)
        self.nemoNormalizeSentence = Self.resolve("nemo_normalize_sentence", from: handle)
        self.nemoNormalizeSentenceWithOptions = Self.resolve(
            "nemo_normalize_sentence_with_options",
            from: handle
        )
        self.nemoNormalizeSentenceWithMaxSpan = Self.resolve(
            "nemo_normalize_sentence_with_max_span",
            from: handle
        )
        self.nemoTnNormalize = Self.resolve("nemo_tn_normalize", from: handle)
        self.nemoTnNormalizeSentence = Self.resolve("nemo_tn_normalize_sentence", from: handle)
        self.nemoFreeString = Self.resolve("nemo_free_string", from: handle)
        self.nemoAddRule = Self.resolve("nemo_add_rule", from: handle)
        self.nemoRemoveRule = Self.resolve("nemo_remove_rule", from: handle)
        self.nemoClearRules = Self.resolve("nemo_clear_rules", from: handle)
        self.nemoRuleCount = Self.resolve("nemo_rule_count", from: handle)
        self.nemoVersion = Self.resolve("nemo_version", from: handle)
        self.isNativeAvailable =
            self.nemoNormalize != nil && self.nemoFreeString != nil && self.nemoVersion != nil
        #endif
    }

    #if !canImport(CNemoTextProcessing)
    private static func resolve<T>(
        _ name: String,
        from handle: UnsafeMutableRawPointer?
    ) -> T? {
        guard let handle, let symbol = dlsym(handle, name) else { return nil }
        return unsafeBitCast(symbol, to: T.self)
    }
    #endif

    // MARK: - Normalization

    /// Normalize spoken-form text to written form (single expression).
    ///
    /// - Parameter input: Spoken-form text from ASR (e.g., "two hundred")
    /// - Returns: Written-form text (e.g., "200"), or original if no normalization applies
    public func normalize(_ input: String) -> String {
        guard let normalize = nemoNormalize, let freeString = nemoFreeString,
            let resultPtr = input.withCString({ normalize($0) })
        else {
            return input
        }
        defer { freeString(resultPtr) }
        return String(cString: resultPtr)
    }

    // MARK: - Text Normalization (written → spoken)

    /// Normalize written-form text to spoken form (single expression), e.g.
    /// `"$5.50"` → `"five dollars fifty cents"`.
    public func tnNormalize(_ input: String) -> String {
        guard let normalize = nemoTnNormalize, let freeString = nemoFreeString,
            let resultPtr = input.withCString({ normalize($0) })
        else {
            return input
        }
        defer { freeString(resultPtr) }
        return String(cString: resultPtr)
    }

    /// Normalize a full sentence to spoken form, rewriting written-form spans
    /// in place (`"I paid $5"` → `"I paid five dollars"`).
    public func tnNormalizeSentence(_ input: String) -> String {
        guard let normalize = nemoTnNormalizeSentence, let freeString = nemoFreeString,
            let resultPtr = input.withCString({ normalize($0) })
        else {
            return input
        }
        defer { freeString(resultPtr) }
        return String(cString: resultPtr)
    }

    /// Normalize a full sentence, replacing spoken-form spans with written form.
    ///
    /// Uses a sliding window to find normalizable spans within the sentence.
    /// Applies NLTagger-based context spotting to avoid false positives on
    /// ambiguous words (e.g., "period" as a noun stays unchanged).
    ///
    /// - Parameter input: Full sentence from ASR
    /// - Returns: Sentence with spoken-form spans replaced
    public func normalizeSentence(_ input: String) -> String {
        guard isNativeAvailable else { return input }
        let (masked, restore) = maskAmbiguousWords(in: input)
        let normalized: String
        if let normalize = nemoNormalizeSentence, let freeString = nemoFreeString,
            let resultPtr = masked.withCString({ normalize($0) })
        {
            defer { freeString(resultPtr) }
            normalized = String(cString: resultPtr)
        } else {
            normalized = normalize(masked)
        }
        return restoreMaskedWords(normalized, restore)
    }

    /// Normalize a full sentence with a configurable max span size.
    ///
    /// - Parameters:
    ///   - input: Full sentence from ASR
    ///   - maxSpanTokens: Maximum consecutive tokens per normalizable span
    /// - Returns: Sentence with spoken-form spans replaced
    public func normalizeSentence(_ input: String, maxSpanTokens: UInt32) -> String {
        guard isNativeAvailable else { return input }
        let (masked, restore) = maskAmbiguousWords(in: input)
        let normalized: String
        if let normalize = nemoNormalizeSentenceWithOptions, let freeString = nemoFreeString,
            let resultPtr = masked.withCString({ normalize($0, 0, maxSpanTokens, 0) })
        {
            defer { freeString(resultPtr) }
            normalized = String(cString: resultPtr)
        } else if let normalize = nemoNormalizeSentenceWithMaxSpan,
            let freeString = nemoFreeString,
            let resultPtr = masked.withCString({ normalize($0, maxSpanTokens) })
        {
            defer { freeString(resultPtr) }
            normalized = String(cString: resultPtr)
        } else if let normalize = nemoNormalizeSentence, let freeString = nemoFreeString,
            let resultPtr = masked.withCString({ normalize($0) })
        {
            defer { freeString(resultPtr) }
            normalized = String(cString: resultPtr)
        } else {
            normalized = normalize(masked)
        }
        return restoreMaskedWords(normalized, restore)
    }

    /// Normalize an ASR result, returning a new result with normalized text.
    ///
    /// - Parameter result: The original ASR result
    /// - Returns: A new ASR result with normalized text
    public func normalize(result: ASRResult) -> ASRResult {
        let normalizedText = normalizeSentence(result.text)

        guard normalizedText != result.text else {
            return result
        }

        return ASRResult(
            text: normalizedText,
            confidence: result.confidence,
            duration: result.duration,
            processingTime: result.processingTime,
            tokenTimings: result.tokenTimings,
            ctcDetectedTerms: result.ctcDetectedTerms,
            ctcAppliedTerms: result.ctcAppliedTerms
        )
    }

    // MARK: - Custom Rules

    /// Add a custom spoken→written normalization rule.
    ///
    /// Custom rules have the highest priority, checked before all built-in taggers.
    /// Matching is case-insensitive on the spoken form.
    ///
    /// - Parameters:
    ///   - spoken: The spoken form to match (e.g., "gee pee tee")
    ///   - written: The written replacement (e.g., "GPT")
    public func addRule(spoken: String, written: String) {
        guard let addRule = nemoAddRule else { return }
        spoken.withCString { spokenPointer in
            written.withCString { writtenPointer in
                addRule(spokenPointer, writtenPointer)
            }
        }
    }

    /// Remove a custom normalization rule.
    ///
    /// - Parameter spoken: The spoken form to remove
    /// - Returns: True if the rule was found and removed
    @discardableResult
    public func removeRule(spoken: String) -> Bool {
        guard let removeRule = nemoRemoveRule else { return false }
        return spoken.withCString { removeRule($0) != 0 }
    }

    /// Clear all custom normalization rules.
    public func clearRules() {
        nemoClearRules?()
    }

    /// The number of custom rules currently registered.
    public var ruleCount: Int {
        Int(nemoRuleCount?() ?? 0)
    }

    // MARK: - Info

    /// The native library version.
    public var version: String? {
        guard let version = nemoVersion, let versionPtr = version() else {
            return nil
        }
        return String(cString: versionPtr)
    }

    // MARK: - NLTagger Context Spotting

    /// Mask ambiguous words that NLTagger identifies as natural language, so the
    /// native normalizer can't rewrite them (e.g. the noun "period" → ".").
    ///
    /// Each protected word is replaced with a unique Private-Use-Area sentinel
    /// character; the native normalizer passes those through unchanged, and the
    /// caller restores them via the returned map. Returns the (possibly)
    /// rewritten string and a sentinel→original map (empty when nothing was
    /// masked).
    private func maskAmbiguousWords(in input: String) -> (masked: String, restore: [Character: String]) {
        let words = input.split(separator: " ", omittingEmptySubsequences: true)

        // Quick check: are there any ambiguous words at all?
        let hasAmbiguous = words.contains { word in
            Self.ambiguousWords.contains(word.lowercased())
        }
        guard hasAmbiguous else {
            return (input, [:])
        }

        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = input

        var wordRanges: [Range<String.Index>] = []
        var searchStart = input.startIndex
        for word in words {
            guard
                let range = input.range(
                    of: String(word),
                    range: searchStart..<input.endIndex
                )
            else {
                return (input, [:])
            }
            wordRanges.append(range)
            searchStart = range.upperBound
        }

        var result: [String] = []
        var restore: [Character: String] = [:]
        // Private Use Area (U+E000…) — never appears in real ASR text and is
        // passed through untouched by the native normalizer.
        var nextSentinel: UInt32 = 0xE000
        for (index, word) in words.enumerated() {
            let wordLower = word.lowercased()

            guard Self.ambiguousWords.contains(wordLower) else {
                result.append(String(word))
                continue
            }

            let wordRange = wordRanges[index]
            let tag = tagger.tag(at: wordRange.lowerBound, unit: .word, scheme: .lexicalClass).0

            // A noun/verb/adjective/adverb in a multi-word sentence is being used
            // as natural language — mask it so the normalizer leaves it alone.
            // Standalone or "other" usage is a potential punctuation command;
            // leave it for the normalizer to process.
            let isNaturalLanguage = tag == .noun || tag == .verb || tag == .adjective || tag == .adverb

            let hasNumericOperands: Bool
            if Self.numericInfixWords.contains(wordLower), index > 0, index + 1 < words.count {
                let previousTag = tagger.tag(
                    at: wordRanges[index - 1].lowerBound,
                    unit: .word,
                    scheme: .lexicalClass
                ).0
                let nextTag = tagger.tag(
                    at: wordRanges[index + 1].lowerBound,
                    unit: .word,
                    scheme: .lexicalClass
                ).0
                hasNumericOperands = previousTag == .number && nextTag == .number
            } else {
                hasNumericOperands = false
            }

            if isNaturalLanguage && !hasNumericOperands && words.count > 1,
                let scalar = UnicodeScalar(nextSentinel)
            {
                let sentinel = Character(scalar)
                nextSentinel += 1
                restore[sentinel] = String(word)
                result.append(String(sentinel))
            } else {
                result.append(String(word))
            }
        }

        return (result.joined(separator: " "), restore)
    }

    /// Restore sentinel characters produced by ``maskAmbiguousWords(in:)``.
    private func restoreMaskedWords(_ text: String, _ restore: [Character: String]) -> String {
        guard !restore.isEmpty else { return text }
        var out = text
        for (sentinel, original) in restore {
            out = out.replacingOccurrences(of: String(sentinel), with: original)
        }
        return out
    }
}
