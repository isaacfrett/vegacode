// Vega design renderer: loads one local HTML page in the OS's own WKWebView (invisible), at the
// viewport sizes and colour schemes a job asks for, runs a measurement script in the page, tabs
// through its focusable elements with real key events, and writes JSON (plus optional PNGs).
// AppKit + WebKit only, no packages. Built and run by src/vega/design_render.py:
//   xcrun swiftc -O -swift-version 6 -strict-concurrency=complete -warnings-as-errors \
//       -parse-as-library DesignRender.swift -o design-render
//   design-render job.json            (the result is one JSON object on stdout)
//
// The job (JSON): {"page": "/abs/root/index.html", "root": "/abs/root", "timeout": 20,
//   "script": "<async function body returning a JSON value>",
//   "focus_script": "<function body: facts about document.activeElement>", "tabs": 40,
//   "renders": [{"width": 390, "height": 844, "dark": false, "snapshot": "/abs/out.png" | null}]}
// The result: {"renders": [{"width", "height", "dark", "measure": <script value>,
//   "focus": [<focus_script value after each Tab>], "snapshot": path | null, "error": text | null}]}
//
// Isolation: the page is a file:// URL with read access to `root` only; a content rule list
// blocks every load that is not file:, data:, blob: or about: (no network at all: fetch, images,
// fonts, websockets); navigations away from `root` are cancelled; no pop-up windows, downloads,
// persistent storage or Web Inspector. Model-written script runs in WebKit's own sandboxed
// WebContent process; this host only reads its JSON results.

import AppKit
import WebKit

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("design-render: \(message)\n".utf8))
    exit(2)
}

struct RenderSpec: Decodable, Sendable {
    let width: Int
    let height: Int
    let dark: Bool
    let snapshot: String?
}

struct Job: Decodable, Sendable {
    let page: String
    let root: String
    let timeout: Double
    let script: String
    let focus_script: String
    let tabs: Int
    let renders: [RenderSpec]
}

/// A borderless window that can still be key, so WebKit treats the page as focused (focus rings
/// and :focus-visible follow real Tab presses).
final class KeyWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

@MainActor
final class Renderer: NSObject, WKNavigationDelegate, WKUIDelegate {
    let job: Job
    let root: URL
    var rules: WKContentRuleList?
    var ruleStore: URL?
    var loaded: CheckedContinuation<String?, Never>?

    init(job: Job) {
        self.job = job
        self.root = URL(fileURLWithPath: job.root, isDirectory: true).standardizedFileURL
    }

    func inRoot(_ url: URL?) -> Bool {
        guard let url else { return false }
        if url.scheme == "about" || url.scheme == "data" || url.scheme == "blob" { return true }
        guard url.isFileURL else { return false }
        let path = url.standardizedFileURL.path
        return path == root.path || path.hasPrefix(root.path.hasSuffix("/") ? root.path : root.path + "/")
    }

    func compileRules() async {
        // Content rule regexes have no disjunction: one ignore rule per allowed scheme.
        let allow = ["file", "data", "blob", "about"].map {
            #"{"trigger":{"url-filter":"^\#($0):"},"action":{"type":"ignore-previous-rules"}}"#
        }
        let list = #"[{"trigger":{"url-filter":".*"},"action":{"type":"block"}},"# + allow.joined(separator: ",") + "]"
        // A store of this process's own (renders run in parallel; the shared default store races),
        // removed again before the process exits.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vega-design-render-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        ruleStore = dir
        guard let store = WKContentRuleListStore(url: dir) else { fail("could not open a content rule list store") }
        rules = try? await store.compileContentRuleList(forIdentifier: "vega-design-render", encodedContentRuleList: list)
        if rules == nil { fail("could not compile the content rule list") }
    }

    func run() async -> [[String: Any]] {
        await compileRules()
        var out: [[String: Any]] = []
        for spec in job.renders { out.append(await render(spec)) }
        if let ruleStore { try? FileManager.default.removeItem(at: ruleStore) }
        return out
    }

    func render(_ spec: RenderSpec) async -> [String: Any] {
        var result: [String: Any] = ["width": spec.width, "height": spec.height, "dark": spec.dark,
                                     "snapshot": NSNull(), "error": NSNull(), "focus": [Any]()]
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        config.preferences.javaScriptCanOpenWindowsAutomatically = false
        config.preferences.isElementFullscreenEnabled = false
        config.preferences.tabFocusesLinks = true  // Tab reaches links, as with full keyboard access
        if let rules { config.userContentController.add(rules) }
        let frame = NSRect(x: 0, y: 0, width: spec.width, height: spec.height)
        // On screen but fully transparent (clicks pass through it): an off-screen window is "occluded",
        // and WebKit then reports the page hidden and unfocused (no :focus-visible, no rAF). The
        // app is never activated, so the owner's frontmost app keeps the keyboard.
        let window = KeyWindow(contentRect: NSRect(x: 0, y: 0, width: spec.width, height: spec.height),
                               styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let web = WKWebView(frame: frame, configuration: config)
        web.navigationDelegate = self
        web.uiDelegate = self
        web.isInspectable = false
        web.appearance = NSAppearance(named: spec.dark ? .darkAqua : .aqua)
        window.appearance = web.appearance
        window.contentView = web
        window.alphaValue = 0
        window.orderFrontRegardless()
        window.makeKey()
        window.makeFirstResponder(web)
        defer { window.orderOut(nil); window.close() }

        let page = URL(fileURLWithPath: job.page).standardizedFileURL
        guard inRoot(page) else {
            result["error"] = "page is outside the root"
            return result
        }
        let error: String? = await withCheckedContinuation { continuation in
            loaded = continuation
            web.loadFileURL(page, allowingReadAccessTo: root)
        }
        if let error {
            result["error"] = "load failed: \(error)"
            return result
        }
        _ = try? await web.callAsyncJavaScript(
            "if (document.fonts) await document.fonts.ready; await new Promise(r => setTimeout(r, 120)); return 1",
            contentWorld: .page)
        do {
            let value = try await web.callAsyncJavaScript(job.script, contentWorld: .defaultClient)
            result["measure"] = value ?? NSNull()
        } catch {
            result["error"] = "measure script failed: \(error.localizedDescription)"
            return result
        }
        var focus: [Any] = []
        for _ in 0..<max(0, job.tabs) {
            press(web, window)
            try? await Task.sleep(for: .milliseconds(40))
            let value = (try? await web.callAsyncJavaScript(job.focus_script, contentWorld: .defaultClient)) ?? nil
            focus.append(value ?? NSNull())
            if let info = value as? [String: Any], info["wrapped"] as? Bool == true { break }
        }
        result["focus"] = focus
        if let path = spec.snapshot {
            result["snapshot"] = await snapshot(web, window, spec, path) ? path : NSNull()
        }
        return result
    }

    /// One real Tab key press (down and up) delivered to the web view.
    func press(_ web: WKWebView, _ window: NSWindow) {
        for type in [NSEvent.EventType.keyDown, .keyUp] {
            guard let event = NSEvent.keyEvent(with: type, location: .zero, modifierFlags: [], timestamp: 0,
                                               windowNumber: window.windowNumber, context: nil, characters: "\t",
                                               charactersIgnoringModifiers: "\t", isARepeat: false, keyCode: 48)
            else { continue }
            if type == .keyDown { web.keyDown(with: event) } else { web.keyUp(with: event) }
        }
    }

    /// The whole page (up to 6000 CSS px tall) as PNG, for human review.
    func snapshot(_ web: WKWebView, _ window: NSWindow, _ spec: RenderSpec, _ path: String) async -> Bool {
        _ = try? await web.callAsyncJavaScript("document.activeElement && document.activeElement.blur(); return 1",
                                               contentWorld: .page)
        let full = (try? await web.callAsyncJavaScript(
            "return Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)",
            contentWorld: .page)) as? Double ?? Double(spec.height)
        let height = CGFloat(min(6000, max(Double(spec.height), full)))
        window.setContentSize(NSSize(width: CGFloat(spec.width), height: height))
        web.frame = NSRect(x: 0, y: 0, width: CGFloat(spec.width), height: height)
        try? await Task.sleep(for: .milliseconds(250))
        let config = WKSnapshotConfiguration()
        config.rect = NSRect(x: 0, y: 0, width: CGFloat(spec.width), height: height)
        config.afterScreenUpdates = true
        guard let image = try? await web.takeSnapshot(configuration: config),
              let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return false }
        return (try? png.write(to: URL(fileURLWithPath: path))) != nil
    }

    // MARK: isolation

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void) {
        if action.shouldPerformDownload { return decisionHandler(.cancel) }
        decisionHandler(inRoot(action.request.url) ? .allow : .cancel)
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping @MainActor (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(response.canShowMIMEType && inRoot(response.response.url) ? .allow : .cancel)
    }

    func webView(_ webView: WKWebView, createWebViewWith config: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? { nil }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { finish(nil) }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: any Error) {
        finish(error.localizedDescription)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
                 withError error: any Error) {
        finish(error.localizedDescription)
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { finish("web content process ended") }

    func finish(_ error: String?) {
        if let continuation = loaded { loaded = nil; continuation.resume(returning: error) }
    }
}

@MainActor func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else {
        fail("result is not JSON")
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

@main
enum Main {
    @MainActor static func main() {
        let args = CommandLine.arguments
        guard args.count == 2 else { fail("usage: design-render <job.json>") }
        guard let data = FileManager.default.contents(atPath: args[1]),
              let job = try? JSONDecoder().decode(Job.self, from: data) else { fail("unreadable job \(args[1])") }
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let limit = job.timeout * Double(max(1, job.renders.count)) + 10
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(limit))
            fail("timed out after \(Int(limit)) s")
        }
        let renderer = Renderer(job: job)
        Task { @MainActor in
            let renders = await renderer.run()
            emit(["renders": renders])
            exit(0)
        }
        app.run()
    }
}
