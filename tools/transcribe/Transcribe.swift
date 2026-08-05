// Word-level timings from the system recogniser, as JSON on stdout.
//
// This is the measurement half of a transcript. It is very good at when a
// word was said and reliably wrong about what some of them were -- product
// names, brands, anything in Taiwanese, anything the microphone caught over
// traffic. Correcting those is a semantic job for a model that can also see
// the frame, and it happens on the Python side; nothing here guesses.
//
// Built once into a binary because the alternative is paying Swift's compile
// on every clip.

import AVFoundation
import Foundation
import Speech

struct Word: Codable {
    let text: String
    let starts_seconds: Double
    let ends_seconds: Double
    let confidence: Double?
}

struct Utterance: Codable {
    let text: String
    let starts_seconds: Double
    let ends_seconds: Double
    let words: [Word]
}

struct Payload: Codable {
    let locale: String
    let duration_seconds: Double
    let utterances: [Utterance]
}

enum Failure: Error, CustomStringConvertible {
    case usage
    case unsupportedLocale(String, [String])
    case noAudio(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: transcribe <audio.wav> [locale]"
        case .unsupportedLocale(let want, let have):
            return "locale \(want) is not supported; installed: \(have.joined(separator: ", "))"
        case .noAudio(let path):
            return "no readable audio in \(path)"
        }
    }
}

@available(macOS 26.0, *)
func transcribe(path: String, locale wanted: String) async throws -> Payload {
    let url = URL(fileURLWithPath: path)
    guard let file = try? AVAudioFile(forReading: url) else {
        throw Failure.noAudio(path)
    }

    let supported = await SpeechTranscriber.supportedLocales
    guard let locale = supported.first(where: {
        $0.identifier(.bcp47).caseInsensitiveCompare(wanted) == .orderedSame
    }) else {
        throw Failure.unsupportedLocale(
            wanted, supported.map { $0.identifier(.bcp47) }
        )
    }

    // Ask for per-run time ranges. Without this the text comes back as one
    // undifferentiated string and a subtitle has nowhere to sit.
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [],
        attributeOptions: [.audioTimeRange, .transcriptionConfidence]
    )

    // The model for a locale may be supported without being present. Asking
    // for it and waiting is the difference between a first run that works
    // and one that fails on a machine that has never used this language.
    if let request = try await AssetInventory.assetInstallationRequest(
        supporting: [transcriber]
    ) {
        try await request.downloadAndInstall()
    }

    let analyzer = SpeechAnalyzer(modules: [transcriber])

    var utterances: [Utterance] = []
    let collecting = Task {
        for try await result in transcriber.results {
            let attributed = result.text
            var words: [Word] = []
            for run in attributed.runs {
                let piece = String(attributed[run.range].characters)
                guard let range = run.audioTimeRange else { continue }
                let trimmed = piece.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { continue }
                words.append(
                    Word(
                        text: trimmed,
                        starts_seconds: range.start.seconds,
                        ends_seconds: (range.start + range.duration).seconds,
                        confidence: run.transcriptionConfidence.map { Double($0) }
                    )
                )
            }
            let text = String(attributed.characters)
                .trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { continue }
            utterances.append(
                Utterance(
                    text: text,
                    starts_seconds: result.range.start.seconds,
                    ends_seconds: (result.range.start + result.range.duration).seconds,
                    words: words
                )
            )
        }
    }

    _ = try await analyzer.analyzeSequence(from: file)
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    try await collecting.value

    return Payload(
        locale: locale.identifier(.bcp47),
        duration_seconds: Double(file.length) / file.fileFormat.sampleRate,
        utterances: utterances
    )
}

@main
struct Main {
    static func main() async {
        do {
            guard #available(macOS 26.0, *) else {
                throw Failure.noAudio("SpeechTranscriber needs macOS 26")
            }
            let arguments = CommandLine.arguments
            guard arguments.count >= 2 else { throw Failure.usage }
            let payload = try await transcribe(
                path: arguments[1],
                locale: arguments.count > 2 ? arguments[2] : "zh-TW"
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(payload)
            FileHandle.standardOutput.write(data)
        } catch {
            FileHandle.standardError.write(
                Data("\(error)\n".utf8)
            )
            exit(1)
        }
    }
}
