// TODO(M5): Bridge between SwiftUI UI and Orion C runtime
// Wraps orion_infer_init / orion_prefill_ane / orion_decode_cpu_step

import Foundation

class ModelRunner: ObservableObject {
    @Published var isGenerating = false
    @Published var metrics: String = ""

    func generate(prompt: String, maxTokens: Int = 128,
                  temperature: Float = 0.8, topP: Float = 0.9) async -> String {
        // TODO(M5): Call Orion C API via bridging header
        return "Not yet implemented"
    }
}
