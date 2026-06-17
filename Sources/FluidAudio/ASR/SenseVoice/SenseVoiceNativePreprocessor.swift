@preconcurrency import CoreML
import Accelerate
import Foundation

struct SenseVoiceNativePreprocessor: Sendable {
    private static let blobHeaderBytes = 64

    private enum BlobOffset {
        static let cmvnInvStd = 64
        static let cmvnNegMean = 2_368
        static let window = 1_259_136
        static let reWeight = 1_900_864
        static let reBias = 2_427_264
        static let imWeight = 2_428_416
        static let melWeight = 2_954_816
        static let melBias = 3_037_120
    }

    private let cmvnInvStd: [Float]
    private let cmvnNegMean: [Float]
    private let window: [Float]
    private let reWeight: [Float]
    private let reBias: [Float]
    private let imWeight: [Float]
    private let melWeight: [Float]
    private let melBias: [Float]

    static func load(from compiledPreprocessorURL: URL) throws -> SenseVoiceNativePreprocessor {
        let weightsURL = compiledPreprocessorURL
            .appendingPathComponent("weights")
            .appendingPathComponent("weight.bin")
        let data = try Data(contentsOf: weightsURL)
        return SenseVoiceNativePreprocessor(
            cmvnInvStd: try readFloat32(count: SenseVoiceConfig.featureDim, offset: BlobOffset.cmvnInvStd, from: data),
            cmvnNegMean: try readFloat32(count: SenseVoiceConfig.featureDim, offset: BlobOffset.cmvnNegMean, from: data),
            window: try readFloat32(count: 400, offset: BlobOffset.window, from: data),
            reWeight: try readPackedDftWeights(offset: BlobOffset.reWeight, from: data),
            reBias: try readFloat32(count: 257, offset: BlobOffset.reBias, from: data),
            imWeight: try readPackedDftWeights(offset: BlobOffset.imWeight, from: data),
            melWeight: try readFloat32(count: 80 * 257, offset: BlobOffset.melWeight, from: data),
            melBias: try readFloat32(count: 80, offset: BlobOffset.melBias, from: data)
        )
    }

    func computeFeatures(audio: [Float]) throws -> MLMultiArray {
        let frameCount = max(0, (audio.count - 400) / 160 + 1)
        let lfrFrameCount = max(1, (frameCount + 5) / 6)
        let features = try MLMultiArray(
            shape: [1, NSNumber(value: lfrFrameCount), NSNumber(value: SenseVoiceConfig.featureDim)],
            dataType: .float32
        )
        let featurePointer = features.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(featurePointer, 0, features.count * MemoryLayout<Float32>.size)
        guard frameCount > 0 else { return features }

        var fbank = [Float](repeating: 0, count: frameCount * 80)
        var frame = [Float](repeating: 0, count: 400)
        var transformed = [Float](repeating: 0, count: 400)
        var power = [Float](repeating: 0, count: 257)
        var mel = [Float](repeating: 0, count: 80)
        let scale = SenseVoiceConfig.waveformScale

        for frameIndex in 0..<frameCount {
            let start = frameIndex * 160
            var mean: Float = 0
            for i in 0..<400 {
                let value = audio[start + i] * scale
                frame[i] = value
                mean += value
            }
            mean /= 400

            var previous = frame[0] - mean
            for i in 0..<400 {
                let centered = frame[i] - mean
                transformed[i] = (centered - 0.970_000_028_610_229_5 * previous) * window[i]
                previous = centered
            }

            computePowerSpectrum(frame: transformed, power: &power)
            computeMel(power: power, mel: &mel)

            let fbankBase = frameIndex * 80
            for m in 0..<80 {
                fbank[fbankBase + m] = logf(max(mel[m], powf(2, -23)))
            }
        }

        let strides = features.strides.map(\.intValue)
        let timeStride = strides.count > 1 ? strides[1] : SenseVoiceConfig.featureDim
        let dimStride = strides.count > 2 ? strides[2] : 1
        for frameIndex in 0..<lfrFrameCount {
            let featureBase = frameIndex * timeStride
            for lfrOffset in 0..<7 {
                let sourceFrame = min(max(frameIndex * 6 + lfrOffset - 3, 0), frameCount - 1)
                let fbankBase = sourceFrame * 80
                let featureOffset = lfrOffset * 80
                for melIndex in 0..<80 {
                    let dim = featureOffset + melIndex
                    let lfrValue = fbank[fbankBase + melIndex]
                    featurePointer[featureBase + dim * dimStride] = (lfrValue + cmvnNegMean[dim]) * cmvnInvStd[dim]
                }
            }
        }

        return features
    }

    private func computePowerSpectrum(frame: [Float], power: inout [Float]) {
        var re = reBias
        var im = [Float](repeating: 0, count: 257)

        reWeight.withUnsafeBufferPointer { reWeightPointer in
            imWeight.withUnsafeBufferPointer { imWeightPointer in
                frame.withUnsafeBufferPointer { framePointer in
                    re.withUnsafeMutableBufferPointer { rePointer in
                        im.withUnsafeMutableBufferPointer { imPointer in
                            vDSP_mmul(
                                reWeightPointer.baseAddress!, 1,
                                framePointer.baseAddress!, 1,
                                rePointer.baseAddress!, 1,
                                257, 1, 400
                            )
                            vDSP_vadd(
                                rePointer.baseAddress!, 1,
                                reBias, 1,
                                rePointer.baseAddress!, 1,
                                vDSP_Length(257)
                            )
                            vDSP_mmul(
                                imWeightPointer.baseAddress!, 1,
                                framePointer.baseAddress!, 1,
                                imPointer.baseAddress!, 1,
                                257, 1, 400
                            )
                        }
                    }
                }
            }
        }

        power.withUnsafeMutableBufferPointer { powerPointer in
            re.withUnsafeBufferPointer { rePointer in
                im.withUnsafeBufferPointer { imPointer in
                    vDSP_vsq(rePointer.baseAddress!, 1, powerPointer.baseAddress!, 1, vDSP_Length(257))
                    vDSP_vma(
                        imPointer.baseAddress!, 1,
                        imPointer.baseAddress!, 1,
                        powerPointer.baseAddress!, 1,
                        powerPointer.baseAddress!, 1,
                        vDSP_Length(257)
                    )
                }
            }
        }
    }

    private func computeMel(power: [Float], mel: inout [Float]) {
        melWeight.withUnsafeBufferPointer { weightPointer in
            power.withUnsafeBufferPointer { powerPointer in
                mel.withUnsafeMutableBufferPointer { melPointer in
                    vDSP_mmul(
                        weightPointer.baseAddress!, 1,
                        powerPointer.baseAddress!, 1,
                        melPointer.baseAddress!, 1,
                        80, 1, 257
                    )
                    vDSP_vadd(
                        melPointer.baseAddress!, 1,
                        melBias, 1,
                        melPointer.baseAddress!, 1,
                        vDSP_Length(80)
                    )
                }
            }
        }
    }

    private static func readFloat32(count: Int, offset: Int, from data: Data) throws -> [Float] {
        let payloadOffset = offset + blobHeaderBytes
        let byteCount = count * MemoryLayout<Float32>.size
        guard data.count >= payloadOffset + byteCount else {
            throw ASRError.processingFailed("SenseVoice preprocessor constants are truncated")
        }

        return data.withUnsafeBytes { rawBuffer in
            let start = rawBuffer.baseAddress!.advanced(by: payloadOffset)
            return (0..<count).map { index in
                start.loadUnaligned(fromByteOffset: index * MemoryLayout<Float32>.size, as: Float32.self)
            }
        }
    }

    private static func readPackedDftWeights(offset: Int, from data: Data) throws -> [Float] {
        let full = try readFloat32(count: 257 * 512, offset: offset, from: data)
        var packed = [Float](repeating: 0, count: 257 * 400)
        for bin in 0..<257 {
            let fullBase = bin * 512
            let packedBase = bin * 400
            for i in 0..<400 {
                packed[packedBase + i] = full[fullBase + i]
            }
        }
        return packed
    }
}
