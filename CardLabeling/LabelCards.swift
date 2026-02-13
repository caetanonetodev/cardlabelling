//
//  LabelCards.swift
//  CardScanner
//
//  Standalone macOS SwiftUI app for viewing and OCR-processing HEIC business cards.
//  Run with: swift LabelCards.swift
//  Or compile: swiftc -o LabelCards LabelCards.swift -framework Cocoa -framework Vision -framework NaturalLanguage -framework SwiftUI -framework UniformTypeIdentifiers
//

import SwiftUI
import AppKit
import Combine
import CoreML
import Vision
import NaturalLanguage
import UniformTypeIdentifiers

// MARK: - Data Models

struct CardContactData {
    var firstName: String?
    var lastName: String?
    var fullName: String?
    var jobTitle: String?
    var company: String?
    var department: String?
    var phoneNumbers: [String] = []
    var phoneLabels: [String] = []
    var emails: [String] = []
    var websites: [String] = []
    var addresses: [String] = []
    var rawText: String = ""
    var confidence: Double = 0.0

    var displayName: String {
        if let fn = fullName, !fn.isEmpty { return fn }
        let parts = [firstName, lastName].compactMap { $0 }
        return parts.isEmpty ? "Unknown Contact" : parts.joined(separator: " ")
    }
}

// MARK: - Annotation Models

/// Available labels for line-level annotation.
/// Phone/email/URL are labeled O since regex handles them.
enum AnnotationLabel: String, CaseIterable, Codable {
    case O = "O"
    case name = "NAME"
    case title = "TITLE"
    case org = "ORG"
    case addr = "ADDR"
    case dept = "DEPT"

    var displayName: String {
        switch self {
        case .O: return "Other"
        case .name: return "Name"
        case .title: return "Job Title"
        case .org: return "Company"
        case .addr: return "Address"
        case .dept: return "Department"
        }
    }

    var color: Color {
        switch self {
        case .O: return .gray
        case .name: return .blue
        case .title: return .green
        case .org: return .orange
        case .addr: return .purple
        case .dept: return .teal
        }
    }
}

/// A single OCR line with its assigned annotation label.
struct AnnotatedLine: Identifiable, Codable {
    let id: UUID
    var text: String
    var label: AnnotationLabel

    init(text: String, label: AnnotationLabel = .O) {
        self.id = UUID()
        self.text = text
        self.label = label
    }
}

/// All annotations for a single business card.
struct AnnotatedCard: Codable {
    let filename: String
    var lines: [AnnotatedLine]
}

/// Top-level export structure.
struct AnnotationExport: Codable {
    let version: String
    let annotationDate: String
    var cards: [AnnotatedCard]

    enum CodingKeys: String, CodingKey {
        case version
        case annotationDate = "annotation_date"
        case cards
    }
}

// MARK: - OCR Service (Vision framework, macOS)

final class CardOCRService {

    func recognizeText(from image: NSImage) async throws -> (text: String, confidence: Double) {
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let cgImage = bitmap.cgImage else {
            throw NSError(domain: "OCR", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid image"])
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US"]
        request.usesLanguageCorrection = true
        request.customWords = [
            "CEO", "CTO", "CFO", "COO", "VP", "Inc", "LLC", "Ltd",
            "Director", "Manager", "Engineer", "Developer", "Designer",
            "Mobile", "Office", "Fax", "Email", "Website", "www"
        ]
        request.minimumTextHeight = 0.03

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        try handler.perform([request])

        guard let observations = request.results, !observations.isEmpty else {
            throw NSError(domain: "OCR", code: 2, userInfo: [NSLocalizedDescriptionKey: "No text detected"])
        }

        var allText = ""
        var totalConfidence: Float = 0
        var count: Float = 0

        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let text = candidate.string
            let conf = candidate.confidence
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let minConf: Float = trimmed.count <= 2 ? 0.7 : 0.5
            if conf >= minConf {
                allText += text + "\n"
                totalConfidence += conf
                count += 1
            }
        }

        guard count > 0 else {
            throw NSError(domain: "OCR", code: 3, userInfo: [NSLocalizedDescriptionKey: "Low confidence"])
        }

        let avg = Double(totalConfidence / count)
        return (allText.trimmingCharacters(in: .whitespacesAndNewlines), avg)
    }
}

// MARK: - WordPiece Tokenizer (DistilBERT-compatible)

/// A minimal WordPiece tokenizer that reads the standard vocab.txt produced by
/// HuggingFace DistilBERT and produces input_ids + attention_mask arrays
/// suitable for CoreML inference.
final class WordPieceTokenizer {

    private let vocab: [String: Int]
    private let idToToken: [Int: String]
    private let unkTokenId: Int
    private let clsTokenId: Int
    private let sepTokenId: Int
    private let padTokenId: Int
    private let maxLength: Int

    init?(vocabURL: URL, maxLength: Int = 128) {
        guard let text = try? String(contentsOf: vocabURL, encoding: .utf8) else { return nil }
        var v: [String: Int] = [:]
        var r: [Int: String] = [:]
        for (i, line) in text.components(separatedBy: .newlines).enumerated() {
            let token = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !token.isEmpty else { continue }
            v[token] = i
            r[i] = token
        }
        self.vocab = v
        self.idToToken = r
        self.unkTokenId = v["[UNK]"] ?? 100
        self.clsTokenId = v["[CLS]"] ?? 101
        self.sepTokenId = v["[SEP]"] ?? 102
        self.padTokenId = v["[PAD]"] ?? 0
        self.maxLength = maxLength
    }

    /// Tokenise a list of pre-split words.  Returns (inputIds, attentionMask, wordIdForEachToken).
    /// `wordIdForEachToken` maps each token position back to its source word index
    /// (nil for [CLS], [SEP], [PAD]).
    func encode(words: [String]) -> (inputIds: [Int32], attentionMask: [Int32], wordIds: [Int?]) {
        var allTokenIds: [Int] = []
        var wordMapping: [Int?] = []

        for (wordIdx, word) in words.enumerated() {
            let subTokens = wordPieceTokenize(word.lowercased())
            for tid in subTokens {
                allTokenIds.append(tid)
                wordMapping.append(wordIdx)
            }
        }

        // Truncate to maxLength - 2 (leave room for [CLS] and [SEP])
        let maxContent = maxLength - 2
        if allTokenIds.count > maxContent {
            allTokenIds = Array(allTokenIds.prefix(maxContent))
            wordMapping = Array(wordMapping.prefix(maxContent))
        }

        // Build final arrays: [CLS] + tokens + [SEP] + [PAD]...
        var inputIds: [Int32] = [Int32(clsTokenId)]
        var attentionMask: [Int32] = [1]
        var finalWordIds: [Int?] = [nil]

        for (i, tid) in allTokenIds.enumerated() {
            inputIds.append(Int32(tid))
            attentionMask.append(1)
            finalWordIds.append(wordMapping[i])
        }

        inputIds.append(Int32(sepTokenId))
        attentionMask.append(1)
        finalWordIds.append(nil)

        // Pad
        while inputIds.count < maxLength {
            inputIds.append(Int32(padTokenId))
            attentionMask.append(0)
            finalWordIds.append(nil)
        }

        return (inputIds, attentionMask, finalWordIds)
    }

    /// Standard WordPiece: greedily match longest subword from the vocabulary.
    private func wordPieceTokenize(_ word: String) -> [Int] {
        var tokens: [Int] = []
        var start = word.startIndex
        var isFirst = true

        while start < word.endIndex {
            var end = word.endIndex
            var matched = false
            while start < end {
                var substr = String(word[start..<end])
                if !isFirst { substr = "##" + substr }
                if let tid = vocab[substr] {
                    tokens.append(tid)
                    start = end
                    matched = true
                    break
                }
                end = word.index(before: end)
            }
            if !matched {
                tokens.append(unkTokenId)
                start = word.index(after: start)
            }
            isFirst = false
        }
        return tokens
    }
}

// MARK: - DistilBERT NER Labels

/// Maps the BIO label indices produced by the CoreML model to entity categories.
/// 11 labels: O + 5 entity types x 2 (B/I).
/// Phone, email, and URL are extracted by regex, not the model.
enum NERLabel: Int, CaseIterable {
    case O = 0
    case bName = 1, iName = 2
    case bTitle = 3, iTitle = 4
    case bOrg = 5, iOrg = 6
    case bAddr = 7, iAddr = 8
    case bDept = 9, iDept = 10

    /// The high-level entity category (ignoring B/I prefix).
    var category: String {
        switch self {
        case .O: return "O"
        case .bName, .iName: return "NAME"
        case .bTitle, .iTitle: return "TITLE"
        case .bOrg, .iOrg: return "ORG"
        case .bAddr, .iAddr: return "ADDR"
        case .bDept, .iDept: return "DEPT"
        }
    }

    /// Whether this is a B- (beginning) tag.
    var isBeginning: Bool {
        switch self {
        case .bName, .bTitle, .bOrg, .bAddr, .bDept: return true
        default: return false
        }
    }
}

// MARK: - DistilBERT Contact Parser (CoreML)

/// Replaces the old heuristic-based CardContactParser with a DistilBERT
/// token-classification model running via CoreML.  The model performs NER
/// over the entire OCR text in a single forward pass, labelling every token
/// as one of: NAME, TITLE, ORG, PHONE, EMAIL, URL, ADDR, or O (outside).
final class DistilBERTContactParser {

    private let model: MLModel
    private let tokenizer: WordPieceTokenizer
    private let maxSeqLen = 128

    init?() {
        // Load vocab.txt
        guard let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "txt"),
              let tok = WordPieceTokenizer(vocabURL: vocabURL, maxLength: 128) else {
            print("[DistilBERTContactParser] Failed to load vocab.txt from bundle")
            return nil
        }
        self.tokenizer = tok

        // Load CoreML model -- Xcode compiles .mlpackage into .mlmodelc at build time
        if let compiledURL = Bundle.main.url(forResource: "BusinessCardNER", withExtension: "mlmodelc") {
            do {
                self.model = try MLModel(contentsOf: compiledURL)
            } catch {
                print("[DistilBERTContactParser] Failed to load compiled model: \(error)")
                return nil
            }
        } else if let packageURL = Bundle.main.url(forResource: "BusinessCardNER", withExtension: "mlpackage") {
            // Fallback: compile from .mlpackage at runtime (shouldn't normally happen)
            do {
                let compiled = try MLModel.compileModel(at: packageURL)
                self.model = try MLModel(contentsOf: compiled)
            } catch {
                print("[DistilBERTContactParser] Failed to compile/load model: \(error)")
                return nil
            }
        } else {
            print("[DistilBERTContactParser] BusinessCardNER model not found in bundle")
            return nil
        }
    }

    // MARK: - Public API

    func parse(ocrText: String) -> CardContactData {
        // Split OCR text into words while preserving original casing
        let words = splitIntoWords(ocrText)
        guard !words.isEmpty else {
            var c = CardContactData()
            c.rawText = ocrText
            return c
        }

        // Tokenise
        let (inputIds, attentionMask, wordIds) = tokenizer.encode(words: words.map(\.text))

        // Run inference
        let labelIndices = predict(inputIds: inputIds, attentionMask: attentionMask)

        // Map token-level predictions back to word-level labels.
        // For each word, take the label of its first subword token.
        var wordLabels: [NERLabel] = Array(repeating: .O, count: words.count)
        var wordSeen = Set<Int>()
        for (tokenIdx, maybeWordId) in wordIds.enumerated() {
            guard let wid = maybeWordId, !wordSeen.contains(wid), wid < words.count else { continue }
            wordSeen.insert(wid)
            if tokenIdx < labelIndices.count {
                wordLabels[wid] = NERLabel(rawValue: labelIndices[tokenIdx]) ?? .O
            }
        }

        // Group consecutive tokens with the same entity category into spans.
        let spans = extractSpans(words: words, labels: wordLabels)

        // Build CardContactData
        var contact = CardContactData()
        contact.rawText = ocrText

        // Populate semantic fields from DistilBERT spans
        for span in spans {
            let text = span.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }

            switch span.category {
            case "NAME":
                if contact.fullName == nil {
                    contact.fullName = text.properlyCapitalized
                    let parts = parseNameComponents(text)
                    contact.firstName = parts.0?.properlyCapitalized
                    contact.lastName = parts.1?.properlyCapitalized
                }
            case "TITLE":
                if contact.jobTitle == nil {
                    contact.jobTitle = text.properlyCapitalized
                }
            case "ORG":
                if contact.company == nil {
                    contact.company = text.properlyCapitalized
                }
            case "ADDR":
                let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !cleaned.isEmpty && !contact.addresses.contains(cleaned) {
                    contact.addresses.append(cleaned)
                }
            case "DEPT":
                if contact.department == nil {
                    contact.department = text.properlyCapitalized
                }
            default:
                break
            }
        }

        // Extract phone, email, URL via regex (not ML -- these are structural patterns)
        contact.phoneNumbers = Self.extractPhoneNumbers(from: ocrText)
        contact.emails = Self.extractEmails(from: ocrText)
        contact.websites = Self.extractURLs(from: ocrText)

        return contact
    }

    // MARK: - Inference

    private func predict(inputIds: [Int32], attentionMask: [Int32]) -> [Int] {
        do {
            let inputIdArray = try MLMultiArray(shape: [1, NSNumber(value: maxSeqLen)], dataType: .int32)
            let maskArray = try MLMultiArray(shape: [1, NSNumber(value: maxSeqLen)], dataType: .int32)

            for i in 0..<maxSeqLen {
                inputIdArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: inputIds[i])
                maskArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: attentionMask[i])
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIdArray),
                "attention_mask": MLFeatureValue(multiArray: maskArray),
            ])

            let output = try model.prediction(from: provider)

            // The model outputs logits of shape [1, seqLen, numLabels]
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                print("[DistilBERTContactParser] No 'logits' in model output")
                return Array(repeating: 0, count: maxSeqLen)
            }

            // Argmax along the label dimension
            let numLabels = NERLabel.allCases.count
            var result: [Int] = []
            for pos in 0..<maxSeqLen {
                var bestLabel = 0
                var bestScore = -Float.infinity
                for label in 0..<numLabels {
                    let val = logits[[0, NSNumber(value: pos), NSNumber(value: label)] as [NSNumber]].floatValue
                    if val > bestScore {
                        bestScore = val
                        bestLabel = label
                    }
                }
                result.append(bestLabel)
            }
            return result
        } catch {
            print("[DistilBERTContactParser] Prediction failed: \(error)")
            return Array(repeating: 0, count: maxSeqLen)
        }
    }

    // MARK: - Span extraction

    private struct EntitySpan {
        let category: String
        let text: String
    }

    private func extractSpans(words: [WordToken], labels: [NERLabel]) -> [EntitySpan] {
        var spans: [EntitySpan] = []
        var currentCategory: String?
        var currentWords: [String] = []

        for (i, label) in labels.enumerated() {
            let cat = label.category
            if cat == "O" {
                if let cc = currentCategory, !currentWords.isEmpty {
                    spans.append(EntitySpan(category: cc, text: currentWords.joined(separator: " ")))
                    currentCategory = nil
                    currentWords = []
                }
                continue
            }

            if label.isBeginning || cat != currentCategory {
                // Start a new span
                if let cc = currentCategory, !currentWords.isEmpty {
                    spans.append(EntitySpan(category: cc, text: currentWords.joined(separator: " ")))
                }
                currentCategory = cat
                currentWords = [words[i].original]
            } else {
                // Continue current span
                currentWords.append(words[i].original)
            }
        }

        // Flush last span
        if let cc = currentCategory, !currentWords.isEmpty {
            spans.append(EntitySpan(category: cc, text: currentWords.joined(separator: " ")))
        }

        return spans
    }

    // MARK: - Text splitting

    private struct WordToken {
        let text: String       // lowercased for tokenizer
        let original: String   // original casing
    }

    private func splitIntoWords(_ text: String) -> [WordToken] {
        // Split on whitespace and newlines, preserving original forms
        let raw = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        return raw.map { WordToken(text: $0, original: $0) }
    }

    // MARK: - Name components

    private func parseNameComponents(_ name: String) -> (String?, String?) {
        let prefixes: Set<String> = ["mr", "mrs", "ms", "dr", "prof", "sir", "miss"]
        var words = name.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        while let first = words.first, prefixes.contains(first.lowercased().replacingOccurrences(of: ".", with: "")) {
            words.removeFirst()
        }
        guard !words.isEmpty else { return (nil, nil) }
        if words.count == 1 { return (words[0], nil) }
        return (words[0], words.dropFirst().joined(separator: " "))
    }

    // MARK: - Regex-based extraction (phone, email, URL)

    private static let phonePatterns = [
        #"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"#,
        #"\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{1,4}[\s.\-]?\d{1,9}"#,
        #"\b\d{10}\b"#
    ]

    private static let emailPattern = #"[A-Z0-9a-z._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,64}"#

    private static let urlPatterns = [
        #"https?://[^\s]+"#,
        #"www\.[^\s]+"#,
        #"\b[a-zA-Z0-9\-]+\.(com|net|org|edu|gov|io|co)\b"#
    ]

    private static func regexMatches(_ pattern: String, in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, options: [], range: range).compactMap { m in
            guard let r = Range(m.range, in: text) else { return nil }
            return String(text[r])
        }
    }

    private static func stripLabels(from text: String, labels: [String]) -> String {
        var cleaned = text
        for label in labels {
            cleaned = cleaned.replacingOccurrences(of: label, with: " ", options: .caseInsensitive)
        }
        return cleaned
    }

    private static func extractPhoneNumbers(from text: String) -> [String] {
        var results: [String] = []
        var seen = Set<String>()
        let cleaned = stripLabels(from: text, labels: [
            "Tel:", "Phone:", "Mob:", "Mobile:", "Fax:", "Cell:", "Telephone:",
            "T:", "M:", "F:", "P:", "C:", "Direct:", "Office:"
        ])
        for pattern in phonePatterns {
            for m in regexMatches(pattern, in: cleaned) {
                let digits = m.filter { $0.isNumber || $0 == "+" }
                if !seen.contains(digits) && digits.filter({ $0.isNumber }).count >= 7 {
                    seen.insert(digits)
                    results.append(m.trimmingCharacters(in: .whitespaces))
                }
            }
        }
        return results
    }

    private static func extractEmails(from text: String) -> [String] {
        var results: [String] = []
        var seen = Set<String>()
        let cleaned = stripLabels(from: text, labels: ["Email:", "E-mail:", "E-Mail:", "Mail:", "E:"])
        for m in regexMatches(emailPattern, in: cleaned) {
            let low = m.lowercased()
            if !seen.contains(low) {
                seen.insert(low)
                results.append(low)
            }
        }
        return results
    }

    private static func extractURLs(from text: String) -> [String] {
        var results: [String] = []
        var seen = Set<String>()
        for pattern in urlPatterns {
            for m in regexMatches(pattern, in: text) {
                if m.contains("@") { continue }
                var normalized = m.trimmingCharacters(in: .whitespaces)
                if !normalized.hasPrefix("http://") && !normalized.hasPrefix("https://") {
                    normalized = normalized.hasPrefix("www.") ? "https://" + normalized : "https://www." + normalized
                }
                while normalized.last?.isPunctuation == true { normalized.removeLast() }
                let low = normalized.lowercased()
                if !seen.contains(low) {
                    seen.insert(low)
                    results.append(low)
                }
            }
        }
        return results
    }
}

// MARK: - String Extension (proper capitalization)

fileprivate extension String {
    var properlyCapitalized: String {
        let trimmed = self.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return trimmed }
        if trimmed == trimmed.uppercased() && trimmed.count <= 5 { return trimmed }
        let needsCap = trimmed == trimmed.lowercased() || trimmed == trimmed.uppercased()
        if needsCap { return trimmed.titleCased }
        return trimmed
    }

    var titleCased: String {
        let lowerWords: Set<String> = ["a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "with"]
        let words = self.components(separatedBy: .whitespaces)
        return words.enumerated().map { (i, w) in
            if w.isEmpty { return w }
            if i == 0 { return w.prefix(1).uppercased() + w.dropFirst().lowercased() }
            if lowerWords.contains(w.lowercased()) { return w.lowercased() }
            return w.prefix(1).uppercased() + w.dropFirst().lowercased()
        }.joined(separator: " ")
    }
}

// MARK: - View Model

@MainActor
final class CardViewerViewModel: ObservableObject {
    @Published var cardPaths: [URL] = []
    @Published var currentIndex: Int = 0
    @Published var currentImage: NSImage?
    @Published var isProcessing: Bool = false
    @Published var rawOCRText: String?
    @Published var parsedContact: CardContactData?
    @Published var errorMessage: String?
    @Published var rotationAngle: Double = 0
    @Published var isLabelMode: Bool = false
    @Published var annotatedLines: [AnnotatedLine] = []

    /// Accumulated annotations keyed by card filename.
    private var allAnnotations: [String: [AnnotatedLine]] = [:]

    private let ocrService = CardOCRService()
    private let parser: DistilBERTContactParser? = DistilBERTContactParser()

    var annotatedCardCount: Int {
        allAnnotations.count
    }

    var cardCount: Int { cardPaths.count }
    var currentCardName: String {
        guard cardPaths.indices.contains(currentIndex) else { return "No cards" }
        return cardPaths[currentIndex].lastPathComponent
    }

    func loadCards() {
        // Look for HEIC files in the app bundle's resource directory
        guard let folder = Bundle.main.resourceURL else {
            errorMessage = "Cannot locate app bundle resources"
            return
        }

        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) else {
            errorMessage = "Cannot read directory: \(folder.path)"
            return
        }

        cardPaths = contents
            .filter { $0.pathExtension.uppercased() == "HEIC" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        if cardPaths.isEmpty {
            errorMessage = "No HEIC files found in \(folder.path)"
        } else {
            loadCurrentImage()
        }
    }

    func goNext() {
        guard currentIndex < cardPaths.count - 1 else { return }
        saveCurrentAnnotations()
        currentIndex += 1
        clearResults()
        loadCurrentImage()
        restoreAnnotationsIfNeeded()
    }

    func goPrevious() {
        guard currentIndex > 0 else { return }
        saveCurrentAnnotations()
        currentIndex -= 1
        clearResults()
        loadCurrentImage()
        restoreAnnotationsIfNeeded()
    }

    private func restoreAnnotationsIfNeeded() {
        guard isLabelMode, cardPaths.indices.contains(currentIndex) else { return }
        let filename = cardPaths[currentIndex].lastPathComponent
        if let saved = allAnnotations[filename] {
            annotatedLines = saved
        } else {
            annotatedLines = []
            // Auto-run OCR in label mode
            processCardForLabeling()
        }
    }

    func rotateLeft() {
        rotationAngle -= 90
    }

    func toggleLabelMode() {
        isLabelMode.toggle()
        if isLabelMode {
            // If we have OCR text, build annotation lines from it
            if let text = rawOCRText {
                loadAnnotationLines(from: text)
            } else if currentImage != nil {
                // Auto-run OCR first
                processCardForLabeling()
            }
        }
    }

    /// Save current annotations and export all as JSON via NSSavePanel.
    func exportAnnotations() {
        saveCurrentAnnotations()

        guard !allAnnotations.isEmpty else {
            errorMessage = "No annotations to export"
            return
        }

        let cards = allAnnotations.map { (filename, lines) in
            AnnotatedCard(filename: filename, lines: lines)
        }.sorted { $0.filename < $1.filename }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let export = AnnotationExport(
            version: "1.0",
            annotationDate: formatter.string(from: Date()),
            cards: cards
        )

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "card_annotations.json"
        panel.title = "Export Annotations"

        let response = panel.runModal()

        guard response == .OK, let url = panel.url else { return }
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(export)
            try data.write(to: url)
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    /// Save the current card's annotations into the accumulator.
    func saveCurrentAnnotations() {
        guard !annotatedLines.isEmpty,
              cardPaths.indices.contains(currentIndex) else { return }
        let filename = cardPaths[currentIndex].lastPathComponent
        allAnnotations[filename] = annotatedLines
    }

    private func processCardForLabeling() {
        guard let image = currentImage else { return }
        isProcessing = true
        errorMessage = nil

        Task {
            do {
                let (text, _) = try await ocrService.recognizeText(from: image)
                rawOCRText = text
                if let parser {
                    parsedContact = parser.parse(ocrText: text)
                }
                loadAnnotationLines(from: text)
            } catch {
                errorMessage = error.localizedDescription
            }
            isProcessing = false
        }
    }

    /// Build annotated lines from OCR text, pre-filling labels from ML predictions.
    private func loadAnnotationLines(from ocrText: String) {
        let filename = cardPaths.indices.contains(currentIndex) ? cardPaths[currentIndex].lastPathComponent : ""

        // Restore previously saved annotations if available
        if let saved = allAnnotations[filename] {
            annotatedLines = saved
            return
        }

        let lines = ocrText.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        // Pre-fill labels using ML model if available
        if let parser, let contact = parsedContact {
            annotatedLines = lines.map { line in
                let label = guessLabel(for: line, contact: contact)
                return AnnotatedLine(text: line, label: label)
            }
        } else {
            annotatedLines = lines.map { AnnotatedLine(text: $0, label: .O) }
        }
    }

    /// Guess the annotation label for an OCR line by matching against parsed contact fields.
    private func guessLabel(for line: String, contact: CardContactData) -> AnnotationLabel {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let lower = trimmed.lowercased()

        // Check structured fields (phone/email/url -> O since regex handles them)
        if lower.contains("@") || trimmed.range(of: #"[A-Za-z0-9._%+\-]+@"#, options: .regularExpression) != nil {
            return .O
        }
        for pattern in [#"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"#, #"\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?"#] {
            if trimmed.range(of: pattern, options: .regularExpression) != nil { return .O }
        }
        if lower.hasPrefix("http") || lower.hasPrefix("www.") { return .O }

        // Match against ML-parsed fields
        if let name = contact.fullName, lower == name.lowercased() { return .name }
        if let title = contact.jobTitle, lower == title.lowercased() { return .title }
        if let company = contact.company, lower == company.lowercased() { return .org }
        if let dept = contact.department, lower == dept.lowercased() { return .dept }
        for addr in contact.addresses {
            if lower.contains(addr.lowercased().prefix(20)) { return .addr }
        }

        return .O
    }

    func processCard() {
        guard let image = currentImage else { return }
        isProcessing = true
        errorMessage = nil

        Task {
            do {
                let (text, _) = try await ocrService.recognizeText(from: image)
                rawOCRText = text
                if let parser {
                    let contact = parser.parse(ocrText: text)
                    parsedContact = contact
                } else {
                    errorMessage = "DistilBERT model not available"
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isProcessing = false
        }
    }

    private func loadCurrentImage() {
        guard cardPaths.indices.contains(currentIndex) else { return }
        let url = cardPaths[currentIndex]
        currentImage = NSImage(contentsOf: url)
        if currentImage == nil {
            errorMessage = "Failed to load image: \(url.lastPathComponent)"
        }
    }

    private func clearResults() {
        rawOCRText = nil
        parsedContact = nil
        errorMessage = nil
        rotationAngle = 0
    }
}

// MARK: - SwiftUI Views

// CardViewerApp removed - using LabelCardsApp as the single @main entry point

struct CardViewerContentView: View {
    @StateObject private var vm = CardViewerViewModel()

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()
            HSplitView {
                cardPanel
                    .frame(minWidth: 400)
                resultsPanel
                    .frame(minWidth: 400)
            }
        }
        .onAppear { vm.loadCards() }
    }

    // MARK: Header

    private var headerBar: some View {
        HStack {
            Text("Card Viewer")
                .font(.title2.bold())

            Spacer()

            if vm.cardCount > 0 {
                Text("\(vm.currentIndex + 1) / \(vm.cardCount)")
                    .font(.headline)
                    .monospacedDigit()
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(vm.currentCardName)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Button(action: vm.toggleLabelMode) {
                Label(
                    vm.isLabelMode ? "Classify Mode" : "Label Mode",
                    systemImage: vm.isLabelMode ? "eye" : "tag"
                )
            }
            .help(vm.isLabelMode ? "Switch to classify mode" : "Switch to annotation mode")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: Card Panel (left)

    private var cardPanel: some View {
        VStack(spacing: 12) {
            if let image = vm.currentImage {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .rotationEffect(.degrees(vm.rotationAngle))
                    .animation(.easeInOut(duration: 0.25), value: vm.rotationAngle)
                    .cornerRadius(8)
                    .shadow(radius: 3)
                    .padding(8)
            } else if let err = vm.errorMessage {
                Text(err)
                    .foregroundColor(.red)
                    .padding()
            } else {
                Text("No card loaded")
                    .foregroundColor(.secondary)
                    .padding()
            }

            Spacer(minLength: 0)

            // Navigation + Process buttons
            HStack(spacing: 16) {
                Button(action: vm.goPrevious) {
                    Label("Previous", systemImage: "chevron.left")
                }
                .disabled(vm.currentIndex <= 0)
                .keyboardShortcut(.leftArrow, modifiers: [])

                Button(action: vm.rotateLeft) {
                    Label("Rotate Left", systemImage: "rotate.left")
                }
                .disabled(vm.currentImage == nil)
                .keyboardShortcut("r", modifiers: [.command])

                Button(action: vm.processCard) {
                    if vm.isProcessing {
                        ProgressView()
                            .controlSize(.small)
                            .padding(.horizontal, 4)
                    } else {
                        Text("Process")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.isProcessing || vm.currentImage == nil)
                .keyboardShortcut(.return, modifiers: [.command])

                Button(action: vm.goNext) {
                    Label("Next", systemImage: "chevron.right")
                }
                .disabled(vm.currentIndex >= vm.cardCount - 1)
                .keyboardShortcut(.rightArrow, modifiers: [])
            }
            .padding(.bottom, 12)
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: Results Panel (right)

    private var resultsPanel: some View {
        Group {
            if vm.isLabelMode {
                annotationPanel
            } else {
                classifyPanel
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
    }

    private var classifyPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if vm.rawOCRText == nil && vm.parsedContact == nil && vm.errorMessage == nil && !vm.isProcessing {
                    VStack(spacing: 8) {
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 40))
                            .foregroundColor(.secondary)
                        Text("Press Process to run OCR")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(.top, 100)
                }

                if vm.isProcessing {
                    VStack(spacing: 8) {
                        ProgressView("Processing...")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 100)
                }

                if let error = vm.errorMessage {
                    GroupBox("Error") {
                        Text(error)
                            .foregroundColor(.red)
                            .textSelection(.enabled)
                    }
                }

                if let contact = vm.parsedContact {
                    classifiedDataSection(contact)
                }

                if let raw = vm.rawOCRText {
                    rawOCRSection(raw)
                }
            }
            .padding(16)
        }
    }

    // MARK: Annotation Panel

    private var annotationPanel: some View {
        VStack(spacing: 0) {
            // Header with export button and progress
            HStack {
                Text("Annotation Mode")
                    .font(.headline)

                Spacer()

                Text("\(vm.annotatedCardCount)/\(vm.cardCount) cards labeled")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Button(action: vm.exportAnnotations) {
                    Label("Export All", systemImage: "square.and.arrow.up")
                }
                .disabled(vm.annotatedCardCount == 0)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            if vm.annotatedLines.isEmpty && !vm.isProcessing {
                VStack(spacing: 8) {
                    Image(systemName: "tag")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("Press Process to run OCR, then label each line")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.top, 80)
            } else if vm.isProcessing {
                VStack(spacing: 8) {
                    ProgressView("Running OCR...")
                }
                .frame(maxWidth: .infinity)
                .padding(.top, 80)
            } else {
                // Annotation rows
                ScrollView {
                    VStack(spacing: 1) {
                        ForEach($vm.annotatedLines) { $line in
                            annotationRow(line: $line)
                        }
                    }
                    .padding(.vertical, 8)
                }
            }

            if let error = vm.errorMessage {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 8)
            }
        }
    }

    private func annotationRow(line: Binding<AnnotatedLine>) -> some View {
        HStack(spacing: 8) {
            Text(line.wrappedValue.text)
                .font(.system(.body, design: .monospaced))
                .lineLimit(2)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            Picker("", selection: line.label) {
                ForEach(AnnotationLabel.allCases, id: \.self) { label in
                    Text(label.displayName)
                        .tag(label)
                }
            }
            .frame(width: 120)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(line.wrappedValue.label.color.opacity(0.12))
    }

    // MARK: Classified data

    private func classifiedDataSection(_ contact: CardContactData) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Text("Classified Card Data")
                    .font(.headline)

                Divider()

                fieldRow(icon: "person.fill", label: "Name", value: contact.displayName)

                if let title = contact.jobTitle {
                    fieldRow(icon: "briefcase.fill", label: "Job Title", value: title)
                }
                if let company = contact.company {
                    fieldRow(icon: "building.2.fill", label: "Company", value: company)
                }
                if let department = contact.department {
                    fieldRow(icon: "folder.fill", label: "Department", value: department)
                }

                if !contact.phoneNumbers.isEmpty {
                    fieldRow(icon: "phone.fill", label: "Phone", value: contact.phoneNumbers.joined(separator: "\n"))
                }
                if !contact.emails.isEmpty {
                    fieldRow(icon: "envelope.fill", label: "Email", value: contact.emails.joined(separator: "\n"))
                }
                if !contact.websites.isEmpty {
                    fieldRow(icon: "globe", label: "Website", value: contact.websites.joined(separator: "\n"))
                }
                if !contact.addresses.isEmpty {
                    fieldRow(icon: "mappin.and.ellipse", label: "Address", value: contact.addresses.joined(separator: "\n"))
                }
            }
            .textSelection(.enabled)
        }
    }

    private func fieldRow(icon: String, label: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon)
                .foregroundColor(.accentColor)
                .frame(width: 20, alignment: .center)
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.body)
            }
        }
    }

    // MARK: Raw OCR

    private func rawOCRSection(_ text: String) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Text("Raw OCR Text")
                    .font(.headline)
                Divider()
                Text(text)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
            }
        }
    }
}

// MARK: - Entry Point (moved to CardLabelingApp.swift)
