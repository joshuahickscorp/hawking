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

func deviceKinds(_ devices: [MLComputeDevice]) -> [String] {
    devices.map(kind)
}

@available(macOS 14.4, *)
func inspectPlan(at inputURL: URL) async -> [String: Any] {
    do {
        // MLComputePlan accepts a compiled .mlmodelc. MLModel.compileModel
        // is also public and lets the same probe accept an uncompiled
        // .mlmodel/.mlpackage when the Apple runtime can compile it.
        let compiledURL: URL
        if inputURL.pathExtension == "mlmodelc" {
            compiledURL = inputURL
        } else {
            compiledURL = try await MLModel.compileModel(at: inputURL)
        }
        let configuration = MLModelConfiguration()
        let plan = try await MLComputePlan.load(
            contentsOf: compiledURL,
            configuration: configuration
        )

        guard case let .program(program) = plan.modelStructure else {
            return [
                "status": "PLAN_LOADED_UNSUPPORTED_MODEL_STRUCTURE",
                "compiled_model": compiledURL.path,
                "model_structure": String(describing: plan.modelStructure),
                "operations": [],
            ]
        }

        var operations: [[String: Any]] = []
        for (functionName, function) in program.functions {
            for (index, operation) in function.block.operations.enumerated() {
                let usage = plan.deviceUsage(for: operation)
                let cost = plan.estimatedCost(of: operation)
                let supported = usage.map { deviceKinds($0.supported) } ?? []
                let preferred = usage.map { kind($0.preferred) }
                operations.append([
                    "function": functionName,
                    "index": index,
                    "operator": operation.operatorName,
                    "supported": supported,
                    "preferred": preferred as Any,
                    "estimated_cost_weight": cost?.weight as Any,
                    "placement_status": usage == nil ? "UNKNOWN" : "PLANNED",
                ])
            }
        }

        return [
            "status": "PLANNED",
            "compiled_model": compiledURL.path,
            "api": "MLComputePlan.load(contentsOf:configuration:)",
            "compile_api": inputURL.pathExtension == "mlmodelc"
                ? "not_needed"
                : "MLModel.compileModel(at:)",
            "model_structure": "MLProgram",
            "operations": operations,
        ]
    } catch {
        return [
            "status": "PLAN_LOAD_FAILED",
            "input_model": inputURL.path,
            "error": "\(error)",
            "operations": [],
        ]
    }
}

@main
struct ANEProbe {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        let out = URL(fileURLWithPath: args.first ?? "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json")
        let modelPath = args.dropFirst().first ?? ProcessInfo.processInfo.environment["HAWKING_ANE_COMPILED_MODEL"]
        let devices = MLComputeDevice.allComputeDevices
        let os = ProcessInfo.processInfo.operatingSystemVersion

        var plan: [String: Any] = [
            "status": "NOT_RUN_NO_COMPILED_MLPROGRAM",
            "api": "MLComputePlan.load(contentsOf:configuration:)",
            "operations": [],
        ]
        if let modelPath, !modelPath.isEmpty {
            if #available(macOS 14.4, *) {
                plan = await inspectPlan(at: URL(fileURLWithPath: modelPath))
            } else {
                plan = [
                    "status": "BLOCKED_OS_TOO_OLD_FOR_MLCOMPUTEPLAN",
                    "api": "MLComputePlan.load(contentsOf:configuration:)",
                    "operations": [],
                ]
            }
        }

        let profile: [String: Any] = [
            "schema": "hawking.apple_ane_device_profile.v1",
            "status": plan["status"] as? String == "PLANNED" ? "PLAN_READY" : "DISCOVERED",
            "public_api_only": true,
            "os": "macOS \(os.majorVersion).\(os.minorVersion).\(os.patchVersion)",
            "coreml_deployment_target": "macOS 14.4+ for MLComputePlan",
            "compute_devices": devices.map { ["kind": kind($0), "description": String(describing: $0)] },
            "supported_compute_devices": deviceKinds(devices),
            "neural_engine_present": devices.contains { if case .neuralEngine = $0 { return true }; return false },
            "mlcomputeplan": plan,
            "claim_boundary": plan["status"] as? String == "PLANNED"
                ? "Public MLComputePlan operation support and preferred placement only; no runtime latency, energy, or Flash residency claim."
                : "Public device discovery only; no ANE operation support, placement, latency, energy, or Flash residency claim.",
        ]

        do {
            try FileManager.default.createDirectory(
                at: out.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try JSONSerialization.data(withJSONObject: profile, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: out)
        } catch {
            fputs("ANE probe could not write \(out.path): \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}
