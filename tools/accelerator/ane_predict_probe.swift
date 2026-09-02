import CoreML
import Foundation

func kind(_ d: MLComputeDevice) -> String {
    switch d {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "NEURAL_ENGINE"
    @unknown default: return "UNKNOWN"
    }
}

func computeUnits(named name: String) -> MLComputeUnits {
    switch name {
    case "cpuOnly": return .cpuOnly
    case "cpuAndGPU": return .cpuAndGPU
    case "cpuAndNeuralEngine": return .cpuAndNeuralEngine
    case "all": return .all
    default: return .all
    }
}

func fill(_ array: MLMultiArray, value: Float16) {
    let n = array.count
    let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: n)
    for i in 0..<n { ptr[i] = value }
}

func makeProvider() throws -> MLDictionaryFeatureProvider {
    let x = try MLMultiArray(shape: [1, 10], dataType: .float16)
    let y = try MLMultiArray(shape: [1, 10], dataType: .float16)
    fill(x, value: 1.0)
    fill(y, value: 2.0)
    return try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: x),
        "y": MLFeatureValue(multiArray: y),
    ])
}

func timeSerial(model: MLModel, provider: MLFeatureProvider, repeats: Int) throws -> [Double] {
    _ = try model.prediction(from: provider)
    var samples: [Double] = []
    samples.reserveCapacity(repeats)
    for _ in 0..<repeats {
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try model.prediction(from: provider)
        samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1_000_000_000)
    }
    return samples
}

func timeConcurrent(url: URL, configuration: MLModelConfiguration, provider: MLFeatureProvider, pair: Int) throws -> [String: Any] {
    let left = try MLModel(contentsOf: url, configuration: configuration)
    let right = try MLModel(contentsOf: url, configuration: configuration)
    _ = try left.prediction(from: provider)
    _ = try right.prediction(from: provider)
    var errors: [String] = []
    let t0 = CFAbsoluteTimeGetCurrent()
    let group = DispatchGroup()
    group.enter()
    DispatchQueue.global(qos: .userInitiated).async {
        do { _ = try left.prediction(from: provider) }
        catch { errors.append("left: \(error)") }
        group.leave()
    }
    group.enter()
    DispatchQueue.global(qos: .userInitiated).async {
        do { _ = try right.prediction(from: provider) }
        catch { errors.append("right: \(error)") }
        group.leave()
    }
    group.wait()
    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1_000_000_000
    return [
        "pair": pair,
        "concurrent_elapsed_ns": elapsed,
        "errors": errors,
        "instances": 2,
        "api": "two MLModel instances, DispatchQueue concurrent prediction(from:)",
    ]
}

@main
struct ANEPredictProbe {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        guard args.count >= 2 else {
            fputs("usage: ane_predict_probe OUT.json MODEL.mlmodelc [cpuOnly|cpuAndGPU|cpuAndNeuralEngine|all] [repeats]\n", stderr)
            Foundation.exit(2)
        }
        let out = URL(fileURLWithPath: args[0])
        let modelURL = URL(fileURLWithPath: args[1])
        let unitsName = args.count > 2 ? args[2] : "all"
        let repeats = args.count > 3 ? (Int(args[3]) ?? 8) : 8
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits(named: unitsName)

        var payload: [String: Any] = [
            "schema": "hawking.apple_ane_predict_probe.v1",
            "public_api_only": true,
            "model": modelURL.path,
            "requested_compute_units": unitsName,
            "requested_compute_units_are_not_placement": true,
            "repeats": repeats,
        ]

        do {
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            let provider = try makeProvider()
            let samples = try timeSerial(model: model, provider: provider, repeats: repeats)
            payload["status"] = "MEASURED"
            payload["predict_api"] = "MLModel.prediction(from:)"
            payload["warm_predict_ns"] = samples
            payload["warm_predict_ns_min"] = samples.min() as Any
            payload["warm_predict_ns_max"] = samples.max() as Any
            payload["warm_predict_ns_mean"] = samples.reduce(0, +) / Double(samples.count)
            payload["output_names"] = model.modelDescription.outputDescriptionsByName.keys.sorted()
            payload["input_names"] = model.modelDescription.inputDescriptionsByName.keys.sorted()
            payload["concurrent"] = try timeConcurrent(
                url: modelURL, configuration: configuration, provider: provider, pair: 2
            )
        } catch {
            payload["status"] = "PREDICT_FAILED"
            payload["error"] = "\(error)"
        }

        do {
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: out)
        } catch {
            fputs("ane_predict_probe could not write \(out.path): \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}
