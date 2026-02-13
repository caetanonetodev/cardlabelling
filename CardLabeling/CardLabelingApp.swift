//
//  CardLabelingApp.swift
//  CardLabeling
//
//  Created by Caetano Spuldaro Neto on 11/02/2026.
//

import SwiftUI

@main
struct CardLabelingApp: App {
    var body: some Scene {
        WindowGroup("Label Cards - Card Viewer") {
            CardViewerContentView()
                .frame(minWidth: 1000, minHeight: 700)
        }
    }
}
