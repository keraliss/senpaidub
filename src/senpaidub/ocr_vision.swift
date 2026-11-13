// OCR helper for burned-in subtitles, using macOS's built-in Vision framework.
// Reads image paths on stdin (one per line); writes one JSON object per image to stdout:
//   {"path": ..., "lines": [{"text", "conf", "x", "y", "w", "h"}, ...]}
// Boxes are normalized to 0...1 with the origin at the bottom-left (Vision's convention).
import Foundation
import ImageIO
import Vision

func recognize(_ url: URL) -> [[String: Any]] {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return [] }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do { try handler.perform([request]) } catch { return [] }
    var lines: [[String: Any]] = []
    for obs in request.results ?? [] {
        guard let best = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox
        lines.append([
            "text": best.string, "conf": Double(best.confidence),
            "x": Double(b.origin.x), "y": Double(b.origin.y),
            "w": Double(b.width), "h": Double(b.height),
        ])
    }
    return lines
}

while let raw = readLine() {
    let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if path.isEmpty { continue }
    let obj: [String: Any] = ["path": path, "lines": recognize(URL(fileURLWithPath: path))]
    if let data = try? JSONSerialization.data(withJSONObject: obj) {
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    }
}
