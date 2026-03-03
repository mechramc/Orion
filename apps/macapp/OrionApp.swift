// TODO(M5): SwiftUI demo app
// - Text input for prompt
// - Generate button → streams tokens
// - Metrics overlay (tokens/sec, prefill ms, ANE utilization, memory)

import SwiftUI

@main
struct OrionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var prompt = ""
    @State private var output = ""

    var body: some View {
        VStack(spacing: 16) {
            Text("Orion — On-Device LLM")
                .font(.title)

            TextField("Enter prompt...", text: $prompt)
                .textFieldStyle(.roundedBorder)

            Button("Generate") {
                // TODO(M5): Call into Orion runtime via ModelRunner
                output = "Not yet implemented"
            }

            ScrollView {
                Text(output)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .frame(minWidth: 600, minHeight: 400)
    }
}
