//
//  快捷键.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/15.
//
// ============================================================
//  文件作用：画布快捷键处理。KeyboardShortcutHandler 以 NSViewRepresentable
//  形式挂载到 CanvasView，通过本地事件监听（NSEvent.addLocalMonitorForEvents）
//  捕获按键，使快捷键与工具栏按钮操作等同（均为已有函数的等效操作）：
//    F2            → 改名（对当前选中节点，与右键「改名」完全一致）
//    空格          → 缩放适配（store.reset()）
//    Command+Z     → 撤销（store.undo()）
//    Command+Alt+Z → 重做（store.redo()）
//    Command+C     → 复制选中节点（含节点间连线）为 JSON 到系统剪贴板
//    Command+V     → 从剪贴板粘贴节点（重建，保留相对位置与连线，可跨项目）
//  文本输入框获得焦点时不拦截按键，避免影响打字。
//  互动文件：被 画布 ui.swift 的 CanvasView 引用（KeyboardShortcutHandler）。
// ============================================================

import SwiftUI
import AppKit

// ============================================================

// MARK: - 画布快捷键处理器

/// 捕获画布快捷键并回调对应操作。放在 CanvasView 的 ZStack 中即可生效。
struct KeyboardShortcutHandler: NSViewRepresentable {
    /// 改名：对当前选中节点进入行内重命名
    var onRename: () -> Void
    /// 缩放适配：重置视图偏移 / 缩放 / 比例
    var onFitZoom: () -> Void
    /// 撤销
    var onUndo: () -> Void
    /// 重做
    var onRedo: () -> Void
    /// 复制选中节点（含节点间连线）到剪贴板
    var onCopy: () -> Void
    /// 从剪贴板粘贴节点
    var onPaste: () -> Void
    /// Delete：删除选中节点 / 解散选中的组（由调用方按选中态分派）
    var onDelete: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSView {
        // 用穿透视图承载键盘监听：不参与鼠标命中测试，绝不拦截画布鼠标操作
        let view = PassthroughView()
        context.coordinator.installMonitor()
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.parent = self
    }

    final class Coordinator {
        var parent: KeyboardShortcutHandler
        var monitor: Any?

        init(_ parent: KeyboardShortcutHandler) {
            self.parent = parent
        }

        func installMonitor() {
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self = self else { return event }
                // 文本输入框（重命名 / 画布名 / 搜索框等）获得焦点时不拦截按键
                if let responder = NSApp.keyWindow?.firstResponder,
                   responder is NSTextView || responder is NSTextField {
                    return event
                }
                let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                let keyCode = event.keyCode

                // Command+Alt+Z：重做（先于 Command+Z 判断，避免被 command 分支吞掉）
                if keyCode == 6, flags.contains(.command), flags.contains(.option) {
                    debugLog("快捷键：Command+Alt+Z 重做")
                    self.parent.onRedo()
                    return nil
                }
                // Command+Z：撤销
                if keyCode == 6, flags.contains(.command) {
                    debugLog("快捷键：Command+Z 撤销")
                    self.parent.onUndo()
                    return nil
                }
                // Command+C：复制选中节点（含节点间连线）
                if keyCode == 8, flags.contains(.command) {
                    debugLog("快捷键：Command+C 复制节点")
                    self.parent.onCopy()
                    return nil
                }
                // Command+V：粘贴节点
                if keyCode == 9, flags.contains(.command) {
                    debugLog("快捷键：Command+V 粘贴节点")
                    self.parent.onPaste()
                    return nil
                }
                // F2：改名（允许 fn 修饰；排除 Command/Option/Control/Shift 真实修饰键，避免被系统功能键 flag 挡住）
                if keyCode == 120, !flags.contains(.command), !flags.contains(.option),
                   !flags.contains(.control), !flags.contains(.shift) {
                    debugLog("快捷键：F2 改名")
                    self.parent.onRename()
                    return nil
                }
                // 空格：缩放适配（无修饰键）
                if keyCode == 49, flags.isEmpty {
                    debugLog("快捷键：空格 缩放适配")
                    self.parent.onFitZoom()
                    return nil
                }
                // Delete（退格 51 / 向前删除 117）：删除选中节点或解散选中组（同样允许 fn 修饰）
                if keyCode == 51 || keyCode == 117, !flags.contains(.command), !flags.contains(.option),
                   !flags.contains(.control), !flags.contains(.shift) {
                    debugLog("快捷键：Delete \(keyCode)")
                    self.parent.onDelete()
                    return nil
                }
                return event
            }
        }

        deinit {
            if let monitor = monitor {
                NSEvent.removeMonitor(monitor)
            }
        }
    }
}

// ============================================================

// MARK: - 穿透视图

/// 不参与鼠标命中测试的透明视图：hitTest 恒返回 nil，
/// 鼠标点击/拖拽全部穿透到下层画布，仅用于承载键盘事件监听。
final class PassthroughView: NSView {
    override func hitTest(_ point: NSPoint) -> NSView? {
        return nil
    }
}

// ============================================================

// MARK: - 剪贴板节点数据（Command+C / Command+V 复制粘贴用）

/// 剪贴板中的节点（id 用 var 以便解码恢复原始 id，用于连线映射）
struct ClipboardNode: Codable {
    var id: UUID
    var type: NodeType
    var title: String
    var subtitle: String
    var needsSupplement: Bool
    var position: CGPoint
    var imageFileName: String?
}

/// 剪贴板中的连线（仅记录 from/to，id 粘贴时重建）
struct ClipboardConnection: Codable {
    var fromID: UUID
    var toID: UUID
}

/// 剪贴板 JSON 整体结构：节点数组 + 节点间连线数组
struct ClipboardCanvasData: Codable {
    var nodes: [ClipboardNode]
    var connections: [ClipboardConnection]
}

