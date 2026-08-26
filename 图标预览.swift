//
//  内存监控和终端反馈.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/18.
//

import SwiftUI

// MARK: - 图标呼吸灯墙（临时测试用：同色系，速度随内存变化）

struct IconBreathPreview: View {
    @State private var usage: Double = 0.3 // 模拟内存占用 0...1

    /// 呼吸速度随内存占用连续变化：内存越高闪得越快
    var breathSpeed: Double { 0.4 + usage * 3.0 }

    /// 候选图标：适合内存监控 / 呼吸灯的符号
    private let icons: [(symbol: String, color: Color, label: String)] = [
        ("memorychip", .cyan, "memorychip"),
        ("memorychip.fill", .cyan, "memorychip.fill"),
        ("cpu", .blue, "cpu"),
        ("cpu.fill", .blue, "cpu.fill"),
        ("bolt.fill", .yellow, "bolt.fill"),
        ("flame.fill", .orange, "flame.fill"),
        ("waveform.path.ecg", .green, "waveform.path.ecg"),
        ("gauge.with.dots.needle.67percent", .pink, "gauge…67percent"),
        ("internaldrive.fill", .teal, "internaldrive.fill"),
        ("thermometer.medium", .red, "thermometer.medium"),
        ("antenna.radiowaves.left.and.right", .purple, "antenna…"),
        ("chart.line.uptrend.xyaxis", .green, "chart…xyaxis"),
        ("sparkles", .yellow, "sparkles"),
        ("sensor.tag.radiowaves", .orange, "sensor.tag…"),
        ("circle.hexagongrid.fill", .indigo, "hexagongrid.fill"),
        ("point.3.connected.trianglepath.dotted", .mint, "point.3…"),
    ]

    private let columns = [
        GridItem(.fixed(84), spacing: 12),
        GridItem(.fixed(84), spacing: 12),
        GridItem(.fixed(84), spacing: 12),
        GridItem(.fixed(84), spacing: 12),
    ]

    var body: some View {
        VStack(spacing: 20) {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 16) {
                    ForEach(icons, id: \.symbol) { item in
                        BreathIconDemo(symbol: item.symbol, color: item.color, speed: breathSpeed, label: item.label)
                    }
                }
                .padding(20)
            }
            .frame(height: 400)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.black.opacity(0.85))
            )

            // 模拟内存滑块
            VStack(spacing: 8) {
                Slider(value: $usage, in: 0...1)
                    .frame(width: 340)
                Text("模拟内存 \(Int(usage * 100))%　呼吸 \(String(format: "%.1f", breathSpeed)) 次/秒")
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.white.opacity(0.8))
            }
            .padding(.horizontal)
        }
        .frame(width: 460, height: 560)
    }
}

/// 单个呼吸灯按钮（同色系演示：颜色固定，呼吸频率可调）
struct BreathIconDemo: View {
    let symbol: String
    let color: Color
    let speed: Double
    let label: String

    var body: some View {
        VStack(spacing: 8) {
            TimelineView(.animation) { context in
                let t = context.date.timeIntervalSinceReferenceDate
                let v = (sin(t * speed * 2 * .pi) + 1) / 2 // 0...1 呼吸曲线
                Button(action: {}) {
                    Image(systemName: symbol)
                        .font(.system(size: 18))
                        .foregroundColor(.white)
                        .frame(width: 48, height: 48)
                        .background(
                            Circle().fill(color.opacity(0.25 + 0.55 * v))
                        )
                        .overlay(
                            Circle().stroke(color.opacity(0.4 + 0.6 * v), lineWidth: 2)
                        )
                        .shadow(color: color.opacity(0.3 + 0.7 * v), radius: 5 + 4 * v)
                }
                .buttonStyle(.plain)
            }
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(.white.opacity(0.7))
                .lineLimit(1)
        }
    }
}

#Preview {
    IconBreathPreview()
}
