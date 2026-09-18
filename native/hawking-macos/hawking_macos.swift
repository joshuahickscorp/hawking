// Hawking's deliberately narrow macOS perception helper.
//
// This process is the stable native owner for TCC-sensitive mechanics.  It has
// no Goal, budget, model, credential, or logical workspace authority.  Hawking
// supplies a versioned JSON-lines request and owns all Goal/lease state.  The
// executable is built with the stable com.hawking.native-helper identity and
// promoted through Hawking's native current/candidate/previous release store.

import AppKit
import ApplicationServices
import AVFoundation
import Foundation
import Speech

let schema = "hawking.macos.helper.v1"
let helperVersion = "0.3.0"

func nowISO8601() -> String {
    ISO8601DateFormatter().string(from: Date())
}

func number(_ value: Any?) -> NSNumber? {
    value as? NSNumber
}

func safeString(_ value: Any?, limit: Int = 512) -> String? {
    guard let value else { return nil }
    let text = String(describing: value).trimmingCharacters(in: .whitespacesAndNewlines)
    return text.isEmpty ? nil : String(text.prefix(limit))
}

func applicationRecord(_ app: NSRunningApplication) -> [String: Any] {
    [
        "pid": app.processIdentifier,
        "name": app.localizedName ?? "",
        "bundle_id": app.bundleIdentifier ?? "",
        "active": app.isActive,
        "hidden": app.isHidden,
        "terminated": app.isTerminated,
    ]
}

func windowRecords() -> [[String: Any]] {
    let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
    let rows = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] ?? []
    return rows.compactMap { row in
        let layer = number(row[kCGWindowLayer as String])?.intValue ?? 0
        guard layer == 0 else { return nil }
        let ownerPID = number(row[kCGWindowOwnerPID as String])?.intValue ?? 0
        let windowID = number(row[kCGWindowNumber as String])?.intValue ?? 0
        guard ownerPID > 0, windowID > 0 else { return nil }
        let bounds = row[kCGWindowBounds as String] as? [String: Any] ?? [:]
        return [
            "window_id": windowID,
            "owner_pid": ownerPID,
            "owner": safeString(row[kCGWindowOwnerName as String]) ?? "",
            "title": safeString(row[kCGWindowName as String]) ?? "",
            "bounds": [
                "x": number(bounds["X"])?.doubleValue ?? 0,
                "y": number(bounds["Y"])?.doubleValue ?? 0,
                "width": number(bounds["Width"])?.doubleValue ?? 0,
                "height": number(bounds["Height"])?.doubleValue ?? 0,
            ],
        ]
    }.prefix(128).map { $0 }
}

func axFocusedApplication(_ app: NSRunningApplication?) -> [String: Any] {
    guard let app else {
        return ["trusted": AXIsProcessTrusted(), "available": false]
    }
    let trusted = AXIsProcessTrusted()
    guard trusted else {
        return [
            "trusted": false,
            "available": false,
            "reason": "accessibility permission is not granted",
        ]
    }
    let element = AXUIElementCreateApplication(app.processIdentifier)
    var role: CFTypeRef?
    var title: CFTypeRef?
    let roleStatus = AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &role)
    let titleStatus = AXUIElementCopyAttributeValue(element, kAXTitleAttribute as CFString, &title)
    return [
        "trusted": true,
        "available": roleStatus == .success || titleStatus == .success,
        "role": role as? String ?? "",
        "title": title as? String ?? "",
    ]
}

func axString(_ element: AXUIElement, _ attribute: String) -> String? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else {
        return nil
    }
    return safeString(value)
}

func axBool(_ element: AXUIElement, _ attribute: String) -> Bool? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else {
        return nil
    }
    return (value as? NSNumber)?.boolValue
}

func axChildren(_ element: AXUIElement) -> [AXUIElement] {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &value) == .success else {
        return []
    }
    return (value as? [AXUIElement]) ?? []
}

func targetString(_ target: [String: Any], _ key: String) -> String? {
    safeString(target[key])
}

func targetMatches(_ element: AXUIElement, _ target: [String: Any]) -> Bool {
    let exact = (target["exact"] as? Bool) ?? true
    func equal(_ actual: String?, _ expected: String?) -> Bool {
        guard let expected, !expected.isEmpty else { return true }
        guard let actual else { return false }
        if exact { return actual == expected }
        return actual.localizedCaseInsensitiveContains(expected)
    }
    guard equal(axString(element, kAXRoleAttribute as String), targetString(target, "role")) else { return false }
    guard equal(axString(element, kAXTitleAttribute as String), targetString(target, "title")) else { return false }
    guard equal(axString(element, kAXDescriptionAttribute as String), targetString(target, "label")) else { return false }
    guard equal(axString(element, "AXIdentifier"), targetString(target, "identifier")) else { return false }
    guard equal(axString(element, kAXValueAttribute as String), targetString(target, "value")) else { return false }
    return true
}

func axRecord(_ element: AXUIElement, path: String) -> [String: Any] {
    var row: [String: Any] = ["path": path]
    if let role = axString(element, kAXRoleAttribute as String) { row["role"] = role }
    if let title = axString(element, kAXTitleAttribute as String) { row["title"] = title }
    if let label = axString(element, kAXDescriptionAttribute as String) { row["label"] = label }
    if let identifier = axString(element, "AXIdentifier") { row["identifier"] = identifier }
    if let value = axString(element, kAXValueAttribute as String) { row["value"] = value }
    if let enabled = axBool(element, kAXEnabledAttribute as String) { row["enabled"] = enabled }
    if let focused = axBool(element, kAXFocusedAttribute as String) { row["focused"] = focused }
    return row
}

func matchingAXElements(
    _ element: AXUIElement,
    target: [String: Any],
    path: String = "0",
    depth: Int = 0,
    maxDepth: Int = 16,
    limit: Int = 16
) -> [(AXUIElement, [String: Any])] {
    if depth > maxDepth || limit <= 0 { return [] }
    var result: [(AXUIElement, [String: Any])] = []
    if targetMatches(element, target) {
        result.append((element, axRecord(element, path: path)))
        if result.count >= limit { return result }
    }
    for (index, child) in axChildren(element).prefix(256).enumerated() {
        let childMatches = matchingAXElements(
            child,
            target: target,
            path: "\(path).\(index)",
            depth: depth + 1,
            maxDepth: maxDepth,
            limit: limit - result.count
        )
        result.append(contentsOf: childMatches)
        if result.count >= limit { break }
    }
    return result
}

func requestedPID(_ request: [String: Any]) -> pid_t? {
    if let value = request["pid"] as? NSNumber, value.int32Value > 0 {
        return pid_t(value.int32Value)
    }
    if let value = request["pid"] as? Int, value > 0 { return pid_t(value) }
    return nil
}

func findSemanticTarget(_ request: [String: Any]) -> [String: Any] {
    guard AXIsProcessTrusted(), let pid = requestedPID(request) else {
        return ["pid": requestedPID(request) ?? 0, "matches": [], "trusted": AXIsProcessTrusted()]
    }
    let target = request["target"] as? [String: Any] ?? [:]
    let limit = min(16, max(1, (request["max_results"] as? NSNumber)?.intValue ?? 8))
    let root = AXUIElementCreateApplication(pid)
    let matches = matchingAXElements(root, target: target, limit: limit)
    return [
        "pid": pid,
        "trusted": true,
        "matches": matches.map { $0.1 },
        "match_count": matches.count,
    ]
}

func pressSemanticTarget(_ request: [String: Any]) -> [String: Any] {
    guard AXIsProcessTrusted(), let pid = requestedPID(request) else {
        return ["pressed": false, "trusted": AXIsProcessTrusted()]
    }
    let target = request["target"] as? [String: Any] ?? [:]
    let matches = matchingAXElements(AXUIElementCreateApplication(pid), target: target, limit: 2)
    guard matches.count == 1 else {
        return ["pressed": false, "match_count": matches.count, "trusted": true]
    }
    let status = AXUIElementPerformAction(matches[0].0, kAXPressAction as CFString)
    return [
        "pressed": status == .success,
        "status": status.rawValue,
        "match_count": matches.count,
        "target": matches[0].1,
        "trusted": true,
    ]
}

func permissionState(_ status: AVAuthorizationStatus) -> String {
    switch status {
    case .authorized: return "READY"
    case .notDetermined: return "NOT_REQUESTED"
    case .denied, .restricted: return "MISSING"
    @unknown default: return "UNKNOWN"
    }
}

func speechPermissionState(_ status: SFSpeechRecognizerAuthorizationStatus) -> String {
    switch status {
    case .authorized: return "READY"
    case .notDetermined: return "NOT_REQUESTED"
    case .denied, .restricted: return "MISSING"
    @unknown default: return "UNKNOWN"
    }
}

func tccObservation(_ state: String, reason: String, kind: String = "tcc") -> [String: Any] {
    [
        "state": state,
        "kind": kind,
        "owner": "com.hawking.native-helper",
        "evidence": ["reason": reason],
    ]
}

func requestedCapabilities(_ request: [String: Any]) -> [String] {
    let values = request["capabilities"] as? [Any] ?? []
    let names = values.compactMap { value -> String? in
        guard let text = value as? String else { return nil }
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return clean.isEmpty ? nil : clean
    }
    if names.isEmpty {
        return [
        "FULL_DISK_ACCESS", "ACCESSIBILITY", "SCREEN_CAPTURE",
        "INPUT_MONITORING", "AUTOMATION", "MICROPHONE", "SPEECH_RECOGNITION",
        ]
    }
    var unique: [String] = []
    for name in names where !unique.contains(name) {
        unique.append(name)
    }
    return unique
}

func permissionObservation(_ capability: String) -> [String: Any] {
    let parts = capability.split(separator: ":", maxSplits: 1).map(String.init)
    let base = parts.first?.uppercased() ?? capability.uppercased()
    switch base {
    case "ACCESSIBILITY":
        let ready = AXIsProcessTrusted()
        return tccObservation(ready ? "READY" : "MISSING", reason: ready ? "AXIsProcessTrusted returned true" : "AXIsProcessTrusted returned false")
    case "SCREEN_CAPTURE":
        let ready = CGPreflightScreenCaptureAccess()
        return tccObservation(ready ? "READY" : "MISSING", reason: ready ? "CGPreflightScreenCaptureAccess returned true" : "Screen Recording grant is missing")
    case "MICROPHONE":
        return tccObservation(permissionState(AVCaptureDevice.authorizationStatus(for: .audio)), reason: "AVCaptureDevice authorizationStatus(.audio)")
    case "SPEECH_RECOGNITION":
        return tccObservation(speechPermissionState(SFSpeechRecognizer.authorizationStatus()), reason: "SFSpeechRecognizer.authorizationStatus")
    case "FULL_DISK_ACCESS":
        // Apple does not expose a public boolean for FDA.  Python performs a
        // real protected-path read probe and is the only layer allowed to
        // convert that evidence to READY.
        return tccObservation("UNKNOWN", reason: "no public Full Disk Access boolean; awaiting protected-path probe")
    case "INPUT_MONITORING":
        // There is no supported public preflight API for this TCC service.
        // Do not collapse it into Accessibility or inspect the TCC database.
        return tccObservation("UNKNOWN", reason: "Input Monitoring has no safe public preflight API")
    case "AUTOMATION":
        let target = parts.count > 1 ? parts[1] : "the named target"
        return tccObservation("UNKNOWN", reason: "Apple Events authorization is target-specific; no universal claim for \(target)")
    default:
        return tccObservation("UNKNOWN", reason: "native helper has no TCC probe for \(capability)")
    }
}

func permissionStatus(_ request: [String: Any]) -> [String: Any] {
    let names = requestedCapabilities(request)
    var values: [String: Any] = [:]
    for name in names {
        values[name] = permissionObservation(name)
    }
    return [
        "schema": "hawking.permissions.v1",
        "operation": "permissions.status",
        "helper_version": helperVersion,
        "observed_at": nowISO8601(),
        "owner": "com.hawking.native-helper",
        "capabilities": values,
    ]
}

func requestPermission(_ request: [String: Any]) -> [String: Any] {
    let names = requestedCapabilities(request)
    guard let capability = names.first else {
        return ["requested": false, "reason": "no capability was supplied"]
    }
    let base = capability.split(separator: ":", maxSplits: 1).first.map(String.init)?.uppercased() ?? capability.uppercased()
    switch base {
    case "ACCESSIBILITY":
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
        return ["requested": true, "capability": capability, "observation": permissionObservation(capability)]
    case "SCREEN_CAPTURE":
        let granted = CGRequestScreenCaptureAccess()
        return ["requested": true, "capability": capability, "observation": permissionObservation(capability), "request_returned": granted]
    case "MICROPHONE":
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        AVCaptureDevice.requestAccess(for: .audio) { value in
            granted = value
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 30.0)
        return ["requested": true, "capability": capability, "observation": permissionObservation(capability), "request_returned": granted]
    case "SPEECH_RECOGNITION":
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        SFSpeechRecognizer.requestAuthorization { status in
            granted = status == .authorized
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 30.0)
        return ["requested": true, "capability": capability, "observation": permissionObservation(capability), "request_returned": granted]
    default:
        return ["requested": false, "capability": capability, "observation": permissionObservation(capability), "reason": "Hawking does not trigger this permission through the native helper"]
    }
}

func observe() -> [String: Any] {
    let workspace = NSWorkspace.shared
    let active = workspace.frontmostApplication
    let applications = workspace.runningApplications
        .filter { !$0.isTerminated && $0.processIdentifier > 0 }
        .map(applicationRecord)
        .sorted { (left, right) in
            let lhs = (left["name"] as? String ?? "").localizedCaseInsensitiveCompare(right["name"] as? String ?? "")
            return lhs == .orderedAscending
        }
    return [
        "schema": schema,
        "operation": "observe",
        "observed_at": nowISO8601(),
        "platform": "macos",
        "applications": applications.prefix(128).map { $0 },
        "windows": windowRecords(),
        "focused_app": active.map(applicationRecord) ?? NSNull(),
        "focused_element": axFocusedApplication(active),
        "accessibility": ["trusted": AXIsProcessTrusted()],
        "screen_capture": ["state": CGPreflightScreenCaptureAccess() ? "READY" : "MISSING", "owner": "com.hawking.native-helper"],
        "ocr": ["state": "NOT_IMPLEMENTED", "reason": "Phase E read-only semantic foundation"],
        "input": ["owner": NSNull(), "state": "NO_UNSCOPED_INPUT", "semantic_action": "AXPress"],
    ]
}

func health() -> [String: Any] {
    [
        "schema": schema,
        "operation": "health",
        "helper_version": helperVersion,
        "platform": "macos",
        "accessibility_trusted": AXIsProcessTrusted(),
        "screen_capture_ready": CGPreflightScreenCaptureAccess(),
        "permission_owner": "com.hawking.native-helper",
        "capabilities": [
            "semantic_observe": true,
            "accessibility_observe": AXIsProcessTrusted(),
            "screen_capture": CGPreflightScreenCaptureAccess(),
            "ocr": false,
            "input": false,
            "semantic_axpress": true,
        ],
    ]
}

func emit(_ value: [String: Any]) {
    do {
        let encoded = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(encoded)
        FileHandle.standardOutput.write(Data([0x0A]))
    } catch {
        let fallback = "{\"schema\":\"\\(schema)\",\"ok\":false,\"error\":\"serialization_failed\"}\\n"
        FileHandle.standardOutput.write(fallback.data(using: .utf8)!)
    }
}

while let line = readLine() {
    guard let data = line.data(using: .utf8),
          let request = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        emit(["schema": schema, "ok": false, "error": "invalid_json"])
        continue
    }
    let operation = request["operation"] as? String ?? ""
    switch operation {
    case "health":
        emit(["schema": schema, "ok": true, "result": health()])
    case "observe":
        emit(["schema": schema, "ok": true, "result": observe()])
    case "find":
        emit(["schema": schema, "ok": true, "result": findSemanticTarget(request)])
    case "press":
        let result = pressSemanticTarget(request)
        if (result["pressed"] as? Bool) == true {
            emit(["schema": schema, "ok": true, "result": result])
        } else {
            emit(["schema": schema, "ok": false, "error": "semantic_axpress_rejected", "result": result])
        }
    case "permissions.status":
        emit(["schema": schema, "ok": true, "result": permissionStatus(request)])
    case "permissions.request":
        emit(["schema": schema, "ok": true, "result": requestPermission(request)])
    default:
        emit([
            "schema": schema,
            "ok": false,
            "error": "unsupported_operation",
            "operation": operation,
        ])
    }
}
